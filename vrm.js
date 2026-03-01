import * as THREE from './lib/three.module.js';
import { GLTFLoader } from './lib/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from './lib/jsm/loaders/FBXLoader.js';
import { OrbitControls } from './lib/jsm/controls/OrbitControls.js';
import { VRMALoader } from './lib/jsm/loaders/VRMALoader.js';
import { VRMLoaderPlugin, VRMUtils } from './lib/three-vrm.module.js';
import { loadBVHAnimation, loadMixamoAnimation } from './animationLoader.js';

import { getRequestHeaders, saveSettings, saveSettingsDebounced, sendMessageAsUser } from '../../../../script.js';
import { getContext, extension_settings, getApiUrl, doExtrasFetch, modules } from '../../../extensions.js';

import {
    MODULE_NAME,
    DEBUG_PREFIX,
    VRM_CANVAS_ID,
    FALLBACK_EXPRESSION,
    ANIMATION_FADE_TIME,
    SPRITE_DIV,
    VN_MODE_DIV,
    HITBOXES,
} from "./constants.js";

import {
    currentChatMembers,
    extractDialogue,
    chunkText,
    extractSentencesWithContext,
    fetchInworldTTS,
    fetchSmallLLMTag
} from './utils.js';

import {
    delay
} from '../../../utils.js';

import {
    animations_files,
    animations_groups 
} from './ui.js';

export {
    loadScene,
    loadAllModels,
    setModel,
    unloadModel,
    getVRM,
    setExpression,
    setMotion,
    updateExpression,
    talk,
    updateModel,
    current_avatars,
    renderer,
    camera,
    VRM_CONTAINER_NAME,
    clearModelCache,
    clearAnimationCache,
    setLight,
    setBackground,
    playTimelineMotions,
    processAndQueueTTS,
    stopTTS,
    blendExpressions
}

const VRM_CONTAINER_NAME = "VRM_CONTAINER";
const VRM_COLLIDER_NAME = "VRM_COLLIDER"

// Avatars
let current_avatars = {} // contain loaded avatar variables

// Caches
let models_cache = {};
let animations_cache = {};
let tts_lips_sync_job_id = 0;

// 3D Scene
let renderer = undefined;
let scene = undefined;
let camera = undefined;
let light = undefined;

// gltf and vrm
let currentInstanceId = 0;
let modelId = 0;
let clock = undefined;
const lookAtTarget = new THREE.Object3D();

let cursorTrackingEnabled = false;
let cursorTarget = new THREE.Object3D();
let blendedTarget = new THREE.Object3D();
let cursorTiltState = {};
let cursorPosition = { x: 0, y: 0 };
let activeNaturalMovements = {};

// Passive listener: only updates math when cursor tracking is ON
window.addEventListener('mousemove', (event) => {
    if (extension_settings.vrm && extension_settings.vrm.cursor_tracking) {
        cursorPosition.x = (event.clientX / window.innerWidth) * 2 - 1;
        cursorPosition.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }
}, { passive: true });

function applyNaturalMovementWithSlerp(vrm, boneName, movementConfig, character, modelId, duration = 12000) {
    const avatar = current_avatars[character];
    if (!avatar || avatar.id !== modelId) return;

    if (!avatar.boneOffsets) avatar.boneOffsets = {};
    
    // Start from current offset if it exists (for smooth chaining), otherwise start clean (identity)
    const startQuat = avatar.boneOffsets[boneName] ? avatar.boneOffsets[boneName].clone() : new THREE.Quaternion();
    avatar.boneOffsets[boneName] = startQuat.clone();

    if (!avatar.boneTweens) avatar.boneTweens = {};

    avatar.boneTweens[boneName] = {
        startTime: Date.now(),
        rampDuration: duration * 0.3,
        holdDuration: duration * 0.4,
        totalDuration: duration,
        startQuat: startQuat,
        targetQuat: new THREE.Quaternion().setFromEuler(new THREE.Euler(movementConfig.x, movementConfig.y, movementConfig.z))
    };
}

function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Helper to apply brief expressions during idle movements
function applyIdleExpression(vrm, character, expressionName, intensity = 0.7, duration = 2000, useClassifiedMapping = true) {
    const avatar = current_avatars[character];
    if (!avatar) return;

    let finalExpression = expressionName;
    let finalIntensity = intensity;

    const isWinking = expressionName === 'blinkLeft' || expressionName === 'blinkRight';
    if (isWinking) {
        avatar.winking = true;
        avatar.customWinking = true;
    }

    if (useClassifiedMapping) {
        const model_path = extension_settings.vrm.character_model_mapping[character];
        if (model_path && extension_settings.vrm.model_settings[model_path]) {
            const modelSettings = extension_settings.vrm.model_settings[model_path];
            if (modelSettings.classify_mapping && modelSettings.classify_mapping[expressionName]) {
                const mapping = modelSettings.classify_mapping[expressionName];
                if (mapping.expression && mapping.expression !== 'none') {
                    finalExpression = mapping.expression;
                }
                if (mapping.intensity !== undefined) {
                    finalIntensity = mapping.intensity;
                }
            }
        }
    }

    const blendShapeMapping = getBlendShapeMapping(character, finalExpression);
    if (blendShapeMapping && blendShapeMapping.blendShapes) {
        applyCustomBlendShapeGroupIdle(vrm, character, finalExpression, blendShapeMapping, finalIntensity, duration, isWinking);
        return;
    }

    // Pass expression to the centralized target array for seamless lerping
    avatar.targetExpressions[finalExpression] = finalIntensity;

    setTimeout(() => {
        if (current_avatars[character]) {
            current_avatars[character].targetExpressions[finalExpression] = 0;
            if (isWinking) {
                current_avatars[character].targetExpressions['blinkLeft'] = 0;
                current_avatars[character].targetExpressions['blinkRight'] = 0;
                current_avatars[character].winking = false;
                current_avatars[character].customWinking = false;
            }
        }
    }, duration * 0.7);
}

// Helper to apply custom blend shape groups during idle animations
function applyCustomBlendShapeGroupIdle(vrm, character, expressionName, blendMapping, intensity = 1.0, duration = 2000, isWinking = false) {
    const avatar = current_avatars[character];
    if (!avatar) return;

    if (isWinking) {
        avatar.winking = true;
        avatar.customWinking = true;
    }

    const blendShapes = blendMapping.blendShapes || {};

    for (const [blendShapeName, weight] of Object.entries(blendShapes)) {
        const adjustedIntensity = Math.min(1.0, Math.max(0.0, weight * intensity));
        avatar.targetExpressions[blendShapeName] = adjustedIntensity;
    }

    setTimeout(() => {
        if (current_avatars[character]) {
            for (const blendShapeName in blendShapes) {
                current_avatars[character].targetExpressions[blendShapeName] = 0;
            }
            if (isWinking) {
                current_avatars[character].targetExpressions['blinkLeft'] = 0;
                current_avatars[character].targetExpressions['blinkRight'] = 0;
                current_avatars[character].winking = false;
                current_avatars[character].customWinking = false;
            }
        }
    }, duration * 0.7);
}

// Helper to apply subtle model Y rotation during idle movements
function applyModelRotation(vrm, character, modelId, targetYaw, duration = 7000) {
    const avatar = current_avatars[character];
    if (!avatar || avatar.id !== modelId) return;

    if (avatar.modelRotationOffset === undefined) avatar.modelRotationOffset = 0;

    avatar.modelRotationTween = {
        startTime: Date.now(),
        rampDuration: duration * 0.3,
        holdDuration: duration * 0.4,
        totalDuration: duration,
        startYaw: avatar.modelRotationOffset,
        targetYaw: targetYaw
    };
}

// Helper to get available blend shape names from VRM model
function getAvailableBlendShapeNames(vrm) {
    if (!vrm || !vrm.blendShapeProxy) return [];

    const blendShapeNames =[];
    const expressionMap = vrm.expressionManager?.expressionMap || {};

    for (const expressionName in expressionMap) {
        blendShapeNames.push(expressionName);
    }

    return blendShapeNames;
}

// Helper to apply custom blend shape mapping
function applyCustomBlendShape(vrm, blendShapeName, intensity = 1.0) {
    if (!vrm || !vrm.expressionManager) return;

    const expressionMap = vrm.expressionManager.expressionMap;
    if (!expressionMap[blendShapeName]) {
        console.debug(DEBUG_PREFIX, 'Blend shape not found:', blendShapeName);
        return;
    }

    vrm.expressionManager.setValue(blendShapeName, intensity);
}

// Helper to apply custom blend shape mapping with multiple blend shapes
function applyCustomBlendShapeGroup(character, vrm, blendShapeGroup, intensity = 1.0) {
    if (!vrm || !vrm.expressionManager) return;

    const model_path = extension_settings.vrm.character_model_mapping[character];
    if (!model_path) return;

    const modelSettings = extension_settings.vrm.model_settings[model_path];
    const blendMapping = modelSettings?.blend_shape_mapping?.[blendShapeGroup];
    
    if (!blendMapping || !blendMapping.blendShapes) return;

    for (const [blendShapeName, weight] of Object.entries(blendMapping.blendShapes)) {
        const adjustedIntensity = Math.min(1.0, Math.max(0.0, weight * intensity));
        applyCustomBlendShape(vrm, blendShapeName, adjustedIntensity);
    }
}

// Helper to get blend shape mapping for an expression name
function getBlendShapeMapping(character, expressionName) {
    const model_path = extension_settings.vrm.character_model_mapping[character];
    if (!model_path) return null;
    
    const modelSettings = extension_settings.vrm.model_settings[model_path];
    if (!modelSettings?.blend_shape_mapping) return null;
    
    return modelSettings.blend_shape_mapping[expressionName] || null;
}

// Helper to reset all blend shapes to 0
function resetAllBlendShapes(vrm) {
    if (!vrm || !vrm.expressionManager) return;

    const expressionMap = vrm.expressionManager.expressionMap;
    for (const expressionName in expressionMap) {
        vrm.expressionManager.setValue(expressionName, 0.0);
    }
}

const NATURAL_MOVEMENTS = {
  slowHeadTurn: {
    type: 'head',
    duration: 12000,
    description: 'slow head turn',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;
      const angleY = (Math.random() * 0.35 + 0.17) * direction;
      const angleX = (Math.random() * 0.16 - 0.08);
      const angleZ = (Math.random() * 0.1 - 0.05) * direction;

      const headConfig = { x: angleX, y: angleY, z: angleZ };
      applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 12000);

      setTimeout(() => {
        const neckConfig = { x: angleX * 0.5, y: angleY * 0.42, z: angleZ * 0.6 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 11800);
      }, 200);

      applyModelRotation(vrm, character, modelId, direction * (Math.random() * 0.1 + 0.08), 10000);
    }
  },
  headTilt: {
    type: 'head',
    duration: 12000,
    description: 'curious head tilt',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;
      const angleZ = (Math.random() * 0.35 + 0.27) * direction;
      const angleX = (Math.random() * 0.12 - 0.06);
      const angleY = (Math.random() * 0.16 - 0.08) * direction;

      const headConfig = { x: angleX, y: angleY, z: angleZ };
      applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 12000);

      setTimeout(() => {
        const neckConfig = { x: angleX * 0.5, y: angleY * 0.35, z: angleZ * 0.57 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 11800);
      }, 200);

      if (Math.random() > 0.7) {
        const winkEye = direction > 0 ? 'blinkLeft' : 'blinkRight';
        setTimeout(() => applyIdleExpression(vrm, character, winkEye, 0.9, 1500), 1000);
      } else if (Math.random() > 0.6) {
        setTimeout(() => applyIdleExpression(vrm, character, 'happy', 0.4, 2000), 500);
      }
    }
  },
  slowGlance: {
    type: 'head',
    duration: 10000,
    description: 'casual glance',
    action: (vrm, character, modelId) => {
      const directionX = Math.random() > 0.5 ? 1 : -1;
      const directionY = Math.random() > 0.5 ? 1 : -1;

      const angleX = (Math.random() * 0.14 + 0.05) * directionX;
      const angleY = (Math.random() * 0.28 + 0.13) * directionY;
      const angleZ = (Math.random() * 0.16 - 0.08);

      applyModelRotation(vrm, character, modelId, directionY * (Math.random() * 0.07 + 0.05), 9000);

      if (Math.random() > 0.5) {
        setTimeout(() => applyIdleExpression(vrm, character, 'surprised', 0.5, 2000), 400);
      }

      const headConfig = { x: angleX, y: angleY, z: angleZ };
      applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 10000);

      setTimeout(() => {
        const neckConfig = { x: angleX * 0.5, y: angleY * 0.54, z: angleZ * 0.5 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 9700);
      }, 300);

      setTimeout(() => {
        const spineConfig = { x: angleX * 0.25, y: angleY * 0.43, z: angleZ * 0.38 };
        applyNaturalMovementWithSlerp(vrm, "spine", spineConfig, character, modelId, 9500);
      }, 500);
    }
  },
  lookAround: {
    type: 'head',
    duration: 19000,
    description: 'looking around',
    action: (vrm, character, modelId) => {
      setTimeout(() => applyModelRotation(vrm, character, modelId, 0.09, 4500), 500);
      setTimeout(() => applyModelRotation(vrm, character, modelId, -0.08, 4500), 7000);

      if (Math.random() > 0.4) {
        setTimeout(() => applyIdleExpression(vrm, character, 'happy', 0.45, 1800), 300);
      }

      const directions =[
        { x: 0.12, y: 0.32, duration: 4000 },
        { x: 0.05, y: 0.08, duration: 3000 },
        { x: 0.1, y: -0.28, duration: 4000 },
        { x: 0.02, y: -0.06, duration: 3000 }
      ];

      let delay = 0;
      directions.forEach((step) => {
          setTimeout(() => {
              applyNaturalMovementWithSlerp(vrm, "head", step, character, modelId, step.duration + 1000);
          }, delay);
          delay += step.duration;
      });
    }
  },
  shoulderShrug: {
    type: 'body',
    duration: 6000,
    description: 'shoulder shrug',
    action: (vrm, character, modelId) => {
        const bothShoulders = Math.random() > 0.6;
        const shrugAmount = Math.random() * 0.12 + 0.06;
        
        const config = { x: -shrugAmount, y: 0, z: 0 };
        
        applyNaturalMovementWithSlerp(vrm, "leftShoulder", config, character, modelId, 6000);
        if (bothShoulders) {
            applyNaturalMovementWithSlerp(vrm, "rightShoulder", config, character, modelId, 6000);
        }
    }
  },
  armStretch: {
    type: 'body',
    duration: 8000,
    description: 'arm stretch',
    action: (vrm, character, modelId) => {
        const side = Math.random() > 0.5 ? "left" : "right";
        
        const stretchConfig = { x: -0.2, y: 0, z: side === "left" ? 0.25 : -0.25 };
        applyNaturalMovementWithSlerp(vrm, `${side}UpperArm`, stretchConfig, character, modelId, 8000);
        
        const elbowConfig = { x: -0.12, y: 0, z: 0 };
        applyNaturalMovementWithSlerp(vrm, `${side}LowerArm`, elbowConfig, character, modelId, 8000);
    }
  },
  weightShift: {
    type: 'body',
    duration: 10000,
    description: 'weight shift with spine twist',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;

      applyModelRotation(vrm, character, modelId, direction * (Math.random() * 0.1 + 0.08), 9000);

      const spineConfig = { x: Math.random() * 0.06 - 0.03, y: (Math.random() * 0.22 + 0.1) * direction, z: (Math.random() * 0.2 + 0.05) * direction };
      applyNaturalMovementWithSlerp(vrm, "spine", spineConfig, character, modelId, 10000);

      setTimeout(() => {
        const chestConfig = { x: Math.random() * 0.04 - 0.02, y: (Math.random() * 0.1 + 0.05) * direction, z: (Math.random() * 0.12 + 0.04) * direction };
        applyNaturalMovementWithSlerp(vrm, "upperChest", chestConfig, character, modelId, 9800);
      }, 200);

      setTimeout(() => {
        const hipsConfig = { x: Math.random() * 0.06 - 0.03, y: -(Math.random() * 0.12 + 0.05) * direction, z: (Math.random() * 0.15 + 0.05) * direction };
        applyNaturalMovementWithSlerp(vrm, "hips", hipsConfig, character, modelId, 9650);
      }, 350);

      if (Math.random() > 0.6) {
        setTimeout(() => applyIdleExpression(vrm, character, 'neutral', 0.5, 1500), 800);
      }
    }
  },
  neckStretch: {
    type: 'neck',
    duration: 10000,
    description: 'neck stretch',
    action: (vrm, character, modelId) => {
      const directionX = Math.random() > 0.5 ? 1 : -1;
      const directionY = Math.random() > 0.5 ? 1 : -1;
      const directionZ = Math.random() > 0.5 ? 1 : -1;

      const neckConfig = { x: (Math.random() * 0.12 + 0.06) * directionX, y: (Math.random() * 0.25 + 0.05) * directionY, z: (Math.random() * 0.4 + 0.15) * directionZ };
      applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 10000);

      setTimeout(() => {
        const headConfig = { x: neckConfig.x * 0.7, y: neckConfig.y * 0.6, z: neckConfig.z * 0.8 };
        applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 9800);
      }, 200);

      if (Math.random() > 0.5) {
        setTimeout(() => applyIdleExpression(vrm, character, 'surprised', 0.55, 2200), 500);
      }
    }
  },
  subtleNod: {
    type: 'head',
    duration: 8000,
    description: 'subtle nod',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;
      const headConfig = { x: Math.random() * 0.14 + 0.08, y: (Math.random() * 0.05) * direction, z: (Math.random() * 0.03) * direction };
      applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 8000);

      setTimeout(() => {
        const neckConfig = { x: headConfig.x * 0.55, y: headConfig.y * 0.6, z: headConfig.z * 0.5 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 7800);
      }, 200);

      if (Math.random() > 0.3) {
        setTimeout(() => applyIdleExpression(vrm, character, 'happy', 0.5, 1500), 1000);
      }
    }
  },
  hipShift: {
    type: 'hips',
    duration: 11000,
    description: 'hip shift with rotation',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;

      applyModelRotation(vrm, character, modelId, direction * (Math.random() * 0.08 + 0.08), 9500);

      const hipConfig = { x: (Math.random() * 0.08 - 0.04), y: (Math.random() * 0.25 + 0.1) * direction, z: (Math.random() * 0.22 + 0.12) * direction };
      applyNaturalMovementWithSlerp(vrm, "hips", hipConfig, character, modelId, 11000);

      setTimeout(() => {
        const chestConfig = { x: (Math.random() * 0.05 - 0.025), y: (Math.random() * 0.08 + 0.04) * direction, z: (Math.random() * 0.08 + 0.04) * direction };
        applyNaturalMovementWithSlerp(vrm, "upperChest", chestConfig, character, modelId, 10750);
      }, 250);

      setTimeout(() => {
        const spineConfig = { x: (Math.random() * 0.08 - 0.04), y: -(Math.random() * 0.15 + 0.08) * direction, z: -(Math.random() * 0.12 + 0.07) * direction };
        applyNaturalMovementWithSlerp(vrm, "spine", spineConfig, character, modelId, 10600);
      }, 400);

      if (Math.random() > 0.6) {
        setTimeout(() => applyIdleExpression(vrm, character, 'surprised', 0.6, 2200), 700);
      }
    }
  },
  feminineHipSway: {
    type: 'hips',
    duration: 14000,
    description: 'feminine hip sway',
    action: (vrm, character, modelId) => {
      const swayAmount = Math.random() * 0.25 + 0.22;
      const direction = Math.random() > 0.5 ? 1 : -1;

      applyModelRotation(vrm, character, modelId, direction * (Math.random() * 0.08 + 0.06), 12000);

      const hipConfig = { x: (Math.random() * 0.08 - 0.04), y: Math.random() * 0.15, z: swayAmount };
      applyNaturalMovementWithSlerp(vrm, "hips", hipConfig, character, modelId, 14000);

      setTimeout(() => {
        const chestConfig = { x: (Math.random() * 0.06 - 0.03), y: -(Math.random() * 0.12 + 0.05), z: -swayAmount * 0.35 };
        applyNaturalMovementWithSlerp(vrm, "upperChest", chestConfig, character, modelId, 13800);
      }, 200);

      setTimeout(() => {
        const spineConfig = { x: (Math.random() * 0.06 - 0.03), y: -(Math.random() * 0.1), z: -swayAmount * 0.52 };
        applyNaturalMovementWithSlerp(vrm, "spine", spineConfig, character, modelId, 13600);
      }, 400);

      setTimeout(() => {
        const neckConfig = { x: (Math.random() * 0.04 - 0.02), y: -(Math.random() * 0.08), z: (Math.random() * 0.1 - 0.05) };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 13400);
      }, 600);

      if (Math.random() > 0.3) {
        setTimeout(() => applyIdleExpression(vrm, character, 'happy', 0.5, 2200), 1200);
      }
    }
  },
  coyHeadTilt: {
    type: 'head',
    duration: 11000,
    description: 'coy head tilt',
    action: (vrm, character, modelId) => {
      const direction = Math.random() > 0.5 ? 1 : -1;
      const headConfig = { x: Math.random() * 0.1 + 0.1, y: (Math.random() * 0.12) * direction, z: -(Math.random() * 0.16 + 0.22) * direction };
      applyNaturalMovementWithSlerp(vrm, "head", headConfig, character, modelId, 11000);

      setTimeout(() => {
        const neckConfig = { x: headConfig.x * 0.53, y: headConfig.y * 0.5, z: headConfig.z * 0.58 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 10800);
      }, 200);

      const expression = Math.random() > 0.3 ? 'relaxed' : 'shy';
      setTimeout(() => applyIdleExpression(vrm, character, expression, 1.0, 2500), 1200);
    }
  },
  chestLift: {
    type: 'chest',
    duration: 9000,
    description: 'chest lift',
    action: (vrm, character, modelId) => {
      const upperChest = vrm.humanoid?.getNormalizedBoneNode("upperChest") || vrm.humanoid?.getNormalizedBoneNode("chest");
      if (!upperChest) return;
      const boneName = upperChest.name;

      const chestConfig = { x: Math.random() * 0.1 + 0.18, y: Math.random() * 0.06 - 0.03, z: Math.random() * 0.05 - 0.025 };
      applyNaturalMovementWithSlerp(vrm, boneName, chestConfig, character, modelId, 9000);

      setTimeout(() => {
        const spineConfig = { x: chestConfig.x * 0.55, y: chestConfig.y * 0.5, z: chestConfig.z * 0.5 };
        applyNaturalMovementWithSlerp(vrm, "spine", spineConfig, character, modelId, 8750);
      }, 250);

      setTimeout(() => {
        const neckConfig = { x: -chestConfig.x * 0.35, y: 0, z: 0 };
        applyNaturalMovementWithSlerp(vrm, "neck", neckConfig, character, modelId, 8600);
      }, 400);

      if (Math.random() > 0.4) {
        const expression = Math.random() > 0.5 ? 'happy' : 'relaxed';
        setTimeout(() => applyIdleExpression(vrm, character, expression, 1.0, 2000), 1800);
      }
    }
  },
};

function triggerRandomNaturalMovement(character, modelId) {
    const avatar = current_avatars[character];
    if (!avatar || avatar.id !== modelId) return;

    const model_path = avatar.model_path;
    const defaultMotion = extension_settings.vrm.model_settings[model_path]?.['animation_default']?.['motion'];
    
    // Normalize motion names by removing file extensions and trailing numbers to compare the animation groups
    let currentMotionGroup = avatar.motion.name;
    if (currentMotionGroup && currentMotionGroup !== "none") {
        currentMotionGroup = currentMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
    }
    
    let defaultMotionGroup = defaultMotion;
    if (defaultMotionGroup && defaultMotionGroup !== "none") {
        defaultMotionGroup = defaultMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
    }

    // Determine if character is truly idle (default animation, not talking, nothing queued)
    const isDefaultMotion = (currentMotionGroup === defaultMotionGroup || avatar.motion.name === "none");
    const isNotTalking = !avatar.isPlayingTts && (avatar.talkEnd || 0) <= Date.now();
    const hasNoQueuedMotions = !(avatar.motionQueue && avatar.motionQueue.length > 0);

    // Only trigger if in idle state
    if (isDefaultMotion && isNotTalking && hasNoQueuedMotions) {
        const movementKeys = Object.keys(NATURAL_MOVEMENTS);
        const randomKey = movementKeys[Math.floor(Math.random() * movementKeys.length)];
        const movement = NATURAL_MOVEMENTS[randomKey];

        movement.action(avatar.vrm, character, modelId);

        // Schedule next movement
        const nextDelay = movement.duration + Math.random() * 5000 + 2000;
        avatar.naturalMovementTimer = setTimeout(() => {
            triggerRandomNaturalMovement(character, modelId);
        }, nextDelay);
    } else {
        // Check again later if a specific animation or TTS is currently playing
        avatar.naturalMovementTimer = setTimeout(() => {
            triggerRandomNaturalMovement(character, modelId);
        }, 3000); // Check again in 3 seconds
    }
}

// background
let background = undefined;

// debug
const gridHelper = new THREE.GridHelper( 20, 20 );
const axesHelper = new THREE.AxesHelper( 10 );

function animate() {
    requestAnimationFrame( animate );
    if (renderer !== undefined && scene !== undefined && camera !== undefined) {
        const deltaTime = clock.getDelta();

        for(const character in current_avatars) {
            const avatar = current_avatars[character];
            const vrm = avatar["vrm"];
            const mixer = avatar["animation_mixer"];

            // 1. RESTORE BACKUPS: Restore bones to their clean state from the previous frame
            // This prevents the infinite twisting/spinning bug on un-keyframed bones
            if (avatar.boneBackups) {
                for (const[boneName, backupQuat] of Object.entries(avatar.boneBackups)) {
                    const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
                    if (bone) {
                        bone.quaternion.copy(backupQuat);
                    }
                }
            }

            // 2. MIXER UPDATE: Apply the base animation (e.g. idle.fbx)
            mixer.update( deltaTime );

            // 3. BUSY CHECK: Determine if we need to interrupt idle movements
            let isBusy = false;
            const model_path = avatar.model_path;
            const defaultMotion = extension_settings.vrm.model_settings[model_path]?.['animation_default']?.['motion'];
            
            let currentMotionGroup = avatar.motion.name;
            if (currentMotionGroup && currentMotionGroup !== "none") {
                currentMotionGroup = currentMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
            }
            let defaultMotionGroup = defaultMotion;
            if (defaultMotionGroup && defaultMotionGroup !== "none") {
                defaultMotionGroup = defaultMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
            }
            
            if (currentMotionGroup !== defaultMotionGroup || avatar.isPlayingTts || (avatar.talkEnd || 0) > Date.now() || (avatar.motionQueue && avatar.motionQueue.length > 0)) {
                isBusy = true;
            }

            // 4. PROCEDURAL TWEENS: Calculate current procedural offsets
            if (avatar.boneTweens && avatar.boneOffsets) {
                const now = Date.now();
                for (const[boneName, tween] of Object.entries(avatar.boneTweens)) {
                    const elapsed = now - tween.startTime;
                    
                    // Gracefully interrupt and fade out if avatar becomes busy
                    if (isBusy && elapsed < tween.rampDuration + tween.holdDuration && !tween.isInterrupting) {
                        tween.isInterrupting = true;
                        tween.rampDuration = Math.min(tween.rampDuration, elapsed);
                        tween.holdDuration = 0;
                        tween.totalDuration = elapsed + 800; // 0.8 seconds to smoothly fade out
                        tween.targetQuat = avatar.boneOffsets[boneName].clone(); // Fade from current exact offset
                    }

                    if (elapsed >= tween.totalDuration) {
                        delete avatar.boneTweens[boneName];
                        delete avatar.boneOffsets[boneName];
                        continue;
                    }

                    if (elapsed < tween.rampDuration) {
                        const t = easeInOutCubic(elapsed / tween.rampDuration);
                        avatar.boneOffsets[boneName].slerpQuaternions(tween.startQuat, tween.targetQuat, t);
                    } else if (elapsed < tween.rampDuration + tween.holdDuration) {
                        avatar.boneOffsets[boneName].copy(tween.targetQuat);
                    } else {
                        const rampDownElapsed = elapsed - tween.rampDuration - tween.holdDuration;
                        const rampDownDuration = tween.totalDuration - tween.rampDuration - tween.holdDuration;
                        const t = rampDownDuration > 0 ? easeInOutCubic(rampDownElapsed / rampDownDuration) : 1;
                        const identity = new THREE.Quaternion();
                        avatar.boneOffsets[boneName].slerpQuaternions(tween.targetQuat, identity, t);
                    }
                }
            }

            // 5. SAVE BACKUPS & APPLY: Save the clean state, then apply our offsets on top
            avatar.boneBackups = {};
            if (avatar.boneOffsets) {
                for (const [boneName, offsetQuat] of Object.entries(avatar.boneOffsets)) {
                    const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
                    if (bone) {
                        // Save the clean bone rotation
                        avatar.boneBackups[boneName] = bone.quaternion.clone();
                        // Apply the procedural rotation on top
                        bone.quaternion.multiply(offsetQuat);
                    }
                }
            }

            // 6. MODEL ROTATION OFFSET
            if (avatar.modelRotationTween !== undefined) {
                const tween = avatar.modelRotationTween;
                const now = Date.now();
                const elapsed = now - tween.startTime;

                if (isBusy && elapsed < tween.rampDuration + tween.holdDuration && !tween.isInterrupting) {
                    tween.isInterrupting = true;
                    tween.rampDuration = Math.min(tween.rampDuration, elapsed);
                    tween.holdDuration = 0;
                    tween.totalDuration = elapsed + 800;
                    tween.targetYaw = avatar.modelRotationOffset || 0;
                }

                if (elapsed >= tween.totalDuration) {
                    avatar.modelRotationOffset = 0;
                    delete avatar.modelRotationTween;
                } else {
                    if (elapsed < tween.rampDuration) {
                        const t = easeInOutCubic(elapsed / tween.rampDuration);
                        avatar.modelRotationOffset = tween.startYaw + (tween.targetYaw - tween.startYaw) * t;
                    } else if (elapsed < tween.rampDuration + tween.holdDuration) {
                        avatar.modelRotationOffset = tween.targetYaw;
                    } else {
                        const rampDownElapsed = elapsed - tween.rampDuration - tween.holdDuration;
                        const rampDownDuration = tween.totalDuration - tween.rampDuration - tween.holdDuration;
                        const t = rampDownDuration > 0 ? easeInOutCubic(rampDownElapsed / rampDownDuration) : 1;
                        avatar.modelRotationOffset = tween.targetYaw * (1 - t);
                    }
                }
                
                const baseRy = Number(extension_settings.vrm.model_settings[model_path]?.['ry'] || 0);
                avatar.objectContainer.rotation.y = baseRy + avatar.modelRotationOffset;
            }

            // 7. EYE DARTS & LOOK-AT LOGIC
            if (avatar.eyeTimer === undefined) {
                avatar.eyeTimer = 0;
                avatar.eyeTargetOffset = new THREE.Vector3(0, 0, 0);
                avatar.eyeCurrentOffset = new THREE.Vector3(0, 0, 0);
                avatar.personalLookAtTarget = new THREE.Object3D();
                scene.add(avatar.personalLookAtTarget);
            }

            avatar.eyeTimer -= deltaTime;
            if (avatar.eyeTimer <= 0) {
                avatar.eyeTimer = 0.5 + Math.random() * 3.0;
                if (Math.random() < 0.3) {
                    avatar.eyeTargetOffset.set(0, 0, 0);
                } else {
                    avatar.eyeTargetOffset.set((Math.random() - 0.5) * 1.5, (Math.random() - 0.5) * 1.0, 0);
                }
            }

            avatar.eyeCurrentOffset.lerp(avatar.eyeTargetOffset, deltaTime * 15.0);

            if (extension_settings.vrm.cursor_tracking) {
                if (!avatar.cursorVec) avatar.cursorVec = new THREE.Vector3();
                if (!avatar.cursorLerpTarget) avatar.cursorLerpTarget = new THREE.Vector3();
                if (!avatar.cameraOffsetVec) avatar.cameraOffsetVec = new THREE.Vector3();
                
                avatar.cursorVec.set(cursorPosition.x, cursorPosition.y, 0.5);
                avatar.cursorVec.unproject(camera);
                avatar.cursorVec.sub(camera.position).normalize().multiplyScalar(5.0).add(camera.position);
                
                avatar.cursorLerpTarget.lerp(avatar.cursorVec, deltaTime * 5.0);
                avatar.personalLookAtTarget.position.copy(avatar.cursorLerpTarget);
                
                avatar.cameraOffsetVec.copy(avatar.eyeCurrentOffset).applyQuaternion(camera.quaternion);
                avatar.personalLookAtTarget.position.add(avatar.cameraOffsetVec);
                
                vrm.lookAt.target = avatar.personalLookAtTarget;

            } else if (extension_settings.vrm.follow_camera) {
                camera.getWorldPosition(avatar.personalLookAtTarget.position);
                const cameraOffset = avatar.eyeCurrentOffset.clone();
                cameraOffset.applyQuaternion(camera.quaternion);
                avatar.personalLookAtTarget.position.add(cameraOffset);
                vrm.lookAt.target = avatar.personalLookAtTarget;
            } else {
                const head = vrm.humanoid.getNormalizedBoneNode('head');
                if (head) {
                    head.getWorldPosition(avatar.personalLookAtTarget.position);
                    const forward = new THREE.Vector3(0, 0, 1);
                    forward.applyQuaternion(avatar.objectContainer.quaternion);
                    avatar.personalLookAtTarget.position.add(forward.multiplyScalar(5.0));
                    
                    const localOffset = avatar.eyeCurrentOffset.clone();
                    localOffset.applyQuaternion(avatar.objectContainer.quaternion);
                    avatar.personalLookAtTarget.position.add(localOffset);
                    vrm.lookAt.target = avatar.personalLookAtTarget;
                } else {
                    vrm.lookAt.target = null;
                }
            }

            // 8. EXPRESSION INTERPOLATION
            if (avatar.targetExpressions) {
                for (const[expr, targetVal] of Object.entries(avatar.targetExpressions)) {
                    if (avatar.currentExpressions[expr] === undefined) {
                        avatar.currentExpressions[expr] = vrm.expressionManager.getValue(expr) || 0.0;
                    }
                    
                    let currentVal = avatar.currentExpressions[expr];
                    if (Math.abs(currentVal - targetVal) > 0.001) {
                        const speed = (expr === 'blink') ? 25.0 : 4.0;
                        currentVal += (targetVal - currentVal) * deltaTime * speed; 
                        
                        if (Math.abs(currentVal - targetVal) < 0.01) {
                            currentVal = targetVal;
                        }

                        avatar.currentExpressions[expr] = currentVal;
                        vrm.expressionManager.setValue(expr, currentVal);
                    }
                }
            }

            // 9. FINAL VRM UPDATE
            // Processes the LookAt offsets and simulates the SpringBones (hair/clothes) over the final combined pose
            vrm.update( deltaTime );
        }
        
        gridHelper.visible = extension_settings.vrm.show_grid;
        axesHelper.visible = extension_settings.vrm.show_grid;

        renderer.render( scene, camera );
    }
}
animate();

async function loadScene() {
    clock = new THREE.Clock();
    current_avatars = {};
    models_cache = {};
    animations_cache = {};
    const instanceId = currentInstanceId + 1;
    currentInstanceId = instanceId;

    // Delete the canvas
    if (document.getElementById(VRM_CANVAS_ID) !== null) {
        document.getElementById(VRM_CANVAS_ID).remove();
        // Hide sprite divs
    }
    
    $('#' + SPRITE_DIV).addClass('vrm-hidden');
    $('#' + VN_MODE_DIV).addClass('vrm-hidden');

    if (!extension_settings.vrm.enabled) {
        $('#' + SPRITE_DIV).removeClass('vrm-hidden');
        $('#' + VN_MODE_DIV).removeClass('vrm-hidden');
        return
    }

    clock.start();

    // renderer
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias : true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.domElement.id = VRM_CANVAS_ID;
    document.body.appendChild( renderer.domElement );

    // camera
    camera = new THREE.PerspectiveCamera( 50.0, window.innerWidth / window.innerHeight, 0.1, 100.0 );
    //const camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 1, 1000 );
    camera.position.set( 0.0, 1.0, 5.0 );

    // camera controls
    //const controls = new OrbitControls( camera, renderer.domElement );
    //controls.screenSpacePanning = true;
    //controls.target.set( 0.0, 1.0, 0.0 );
    //controls.update();

    // scene
    scene = new THREE.Scene();
    
    // Grid debuging helpers
    scene.add( gridHelper );
    scene.add( axesHelper );
    gridHelper.visible = extension_settings.vrm.show_grid;
    axesHelper.visible = extension_settings.vrm.show_grid;

    // light
    light = new THREE.DirectionalLight();
    light.position.set( 1.0, 1.0, 1.0 ).normalize();
    setLight(extension_settings.vrm.light_color, extension_settings.vrm.light_intensity);
    scene.add( light );

    // lookat target
    camera.add( lookAtTarget );

    //current_characters = currentChatMembers();
    //await loadAllModels(current_characters);

    //console.debug(DEBUG_PREFIX,"DEBUG",renderer);
}

async function loadAllModels(current_characters) {
    // Unload models
    for(const character in current_avatars) {
        await unloadModel(character);
    }

    if (extension_settings.vrm.enabled) {
        // Load new characters models
        for(const character of current_characters) {
            const model_path = extension_settings.vrm.character_model_mapping[character];
            if (model_path !== undefined) {
                console.debug(DEBUG_PREFIX,"Loading VRM model of",character,":",model_path);
                await setModel(character,model_path);
            }
        }
    }
}

async function setModel(character,model_path) {
    let model;
    // Model is cached
    if (models_cache[model_path] !== undefined) {
        model = models_cache[model_path];
        await initModel(model);
        console.debug(DEBUG_PREFIX,"Model loaded from cache:",model_path);
    }
    else {
        model = await loadModel(model_path);
    }

    await unloadModel(character);

    // Error occured
    if (model === null) {
        extension_settings.vrm.character_model_mapping[character] = undefined;
        return;
    }

    // Set as character model and start animations
    modelId++;
    current_avatars[character] = model;
    current_avatars[character]["id"] = modelId;
    current_avatars[character]["objectContainer"].name = VRM_CONTAINER_NAME+"_"+character;
    current_avatars[character]["collider"].name = VRM_COLLIDER_NAME+"_"+character;

    // Load default expression/motion
    const expression = extension_settings.vrm.model_settings[model_path]['animation_default']['expression'];
    const motion =  extension_settings.vrm.model_settings[model_path]['animation_default']['motion'];

    if (expression !== undefined && expression != "none") {
        console.debug(DEBUG_PREFIX,"Set default expression to",expression);
        await setExpression(character, expression);
    }
    if (motion !== undefined && motion != "none") {
        console.debug(DEBUG_PREFIX,"Set default motion to",motion);
        await setMotion(character, motion, true);
    }

    if (extension_settings.vrm.blink)
        blink(character, modelId);
    textTalk(character, modelId);
    current_avatars[character].naturalMovementTimer = setTimeout(() => {
        triggerRandomNaturalMovement(character, modelId);
    }, 3000);
    current_avatars[character]["objectContainer"].visible = true;
    current_avatars[character]["collider"].visible = extension_settings.vrm.show_grid;
    
    scene.add(current_avatars[character]["objectContainer"]);
    scene.add(current_avatars[character]["collider"]);
    for(const hitbox in current_avatars[character]["hitboxes"])
        scene.add(current_avatars[character]["hitboxes"][hitbox]["offsetContainer"]);
}

async function unloadModel(character) {
    // unload existing model
    if (current_avatars[character] !== undefined) {
        console.debug(DEBUG_PREFIX,"Unloading avatar of",character);
        const container = current_avatars[character]["objectContainer"];
        const collider = current_avatars[character]["collider"];

        scene.remove(scene.getObjectByName(container.name));
        scene.remove(scene.getObjectByName(collider.name));
        for(const hitbox in current_avatars[character]["hitboxes"]) {
            console.debug(DEBUG_PREFIX,"REMOVING",current_avatars[character]["hitboxes"][hitbox]["offsetContainer"])
            scene.remove(scene.getObjectByName(current_avatars[character]["hitboxes"][hitbox]["offsetContainer"].name));
        }

        // Remove personal look at target to prevent memory leaks
        if (current_avatars[character].personalLookAtTarget) {
            scene.remove(current_avatars[character].personalLookAtTarget);
        }

        // unload animations
        current_avatars[character]["animation_mixer"].stopAllAction();
        if (current_avatars[character]["motion"]["animation"]  !== null) {
            current_avatars[character]["motion"]["animation"].stop();
            current_avatars[character]["motion"]["animation"].terminated = true;
            current_avatars[character]["motion"]["animation"] = null;
        }

        // Kill the natural movement loop to prevent memory leaks
        if (current_avatars[character].naturalMovementTimer) {
            clearTimeout(current_avatars[character].naturalMovementTimer);
        }

        delete current_avatars[character];

        container.visible = false;
        collider.visible = false;
        if (!extension_settings.vrm.models_cache) {
            await container.traverse(obj => obj.dispose?.());
            await collider.traverse(obj => obj.dispose?.());
        }
    }
}

async function loadModel(model_path) { // Only cache the model if character=null
    // gltf and vrm
    const loader = new GLTFLoader();
    loader.crossOrigin = 'anonymous';

    loader.register( ( parser ) => {
        return new VRMLoaderPlugin( parser );
    } );

    let gltf;
    try {
        gltf = await loader.loadAsync(model_path,
            // called after loaded
            () => {
                console.debug(DEBUG_PREFIX,"Finished loading",model_path);
            },
            // called while loading is progressing
            ( progress ) => {
                const percent = Math.round(100.0 * ( progress.loaded / progress.total ));
                console.debug(DEBUG_PREFIX, 'Loading model...', percent, '%');
                $("#vrm_model_loading_percent").text(percent);
            },
            // called when loading has errors
            ( error ) => {
                console.debug(DEBUG_PREFIX,"Error when loading",model_path,":",error)
                toastr.error('Wrong avatar file:'+model_path, DEBUG_PREFIX + ' cannot load', { timeOut: 10000, extendedTimeOut: 20000, preventDuplicates: true });
                return;
            }
        );
    }
    catch (error) {
        console.debug(DEBUG_PREFIX,"Error when loading",model_path,":",error)
        toastr.error('Wrong avatar file:'+model_path, DEBUG_PREFIX + ' cannot load', { timeOut: 10000, extendedTimeOut: 20000, preventDuplicates: true });
        return null;
    }

    const vrm = gltf.userData.vrm;
    const vrmHipsY = vrm.humanoid?.getNormalizedBoneNode( 'hips' ).position.y;
    const vrmRootY = vrm.scene.position.y;
    const hipsHeight = Math.abs( vrmHipsY - vrmRootY ); // Used for offset center rotation and animation scaling

    // calling these functions greatly improves the performance
    VRMUtils.removeUnnecessaryVertices( gltf.scene );
    VRMUtils.removeUnnecessaryJoints( gltf.scene );

    // Disable frustum culling
    vrm.scene.traverse( ( obj ) => {
        obj.frustumCulled = false;
    } );

    // un-T-pose
    vrm.springBoneManager.reset();
    if (vrm.meta?.metaVersion === '1') {
        vrm.humanoid.getNormalizedBoneNode("rightUpperArm").rotation.z = -250;
        vrm.humanoid.getNormalizedBoneNode("rightLowerArm").rotation.z = 0.2;
        vrm.humanoid.getNormalizedBoneNode("leftUpperArm").rotation.z = 250;
        vrm.humanoid.getNormalizedBoneNode("leftLowerArm").rotation.z = -0.2;
    }
    else {
        vrm.humanoid.getNormalizedBoneNode("rightUpperArm").rotation.z = 250;
        vrm.humanoid.getNormalizedBoneNode("rightLowerArm").rotation.z = -0.2;
        vrm.humanoid.getNormalizedBoneNode("leftUpperArm").rotation.z = -250;
        vrm.humanoid.getNormalizedBoneNode("leftLowerArm").rotation.z = 0.2;
    }

    // Add vrm to scene
    VRMUtils.rotateVRM0(vrm); // rotate if the VRM is VRM0.0
    const scale = extension_settings.vrm.model_settings[model_path]["scale"];
    // Create a group to set model center as rotation/scaling origin
    const object_container = new THREE.Group(); // First container to scale/position center model
    object_container.visible = false;
    object_container.name = VRM_CONTAINER_NAME;
    object_container.model_path = model_path; // link to character for mouse controls
    object_container.scale.set(scale,scale,scale);
    object_container.position.y = 0.5; // offset to center model
    const verticalOffset = new THREE.Group(); // Second container to rotate center model
    verticalOffset.position.y = -hipsHeight; // offset model for rotate on "center"
    verticalOffset.add(vrm.scene)
    object_container.add(verticalOffset);
    //object_container.parent = scene;
    
    // Collider used to detect mouse click
    const boundingBox = new THREE.Box3(new THREE.Vector3(-0.5,-1.0,-0.5), new THREE.Vector3(0.5,1.0,0.5));
    const dimensions = new THREE.Vector3().subVectors( boundingBox.max, boundingBox.min );
    // make a BoxGeometry of the same size as Box3
    const boxGeo = new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z);
    // move new mesh center so it's aligned with the original object
    const matrix = new THREE.Matrix4().setPosition(dimensions.addVectors(boundingBox.min, boundingBox.max).multiplyScalar( 0.5 ));
    boxGeo.applyMatrix4(matrix);
    // make a mesh
    const collider = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({
        visible: true,
        side: THREE.BackSide,
        wireframe: true,
        color:0xffff00
    }));
    collider.name = VRM_COLLIDER_NAME;
    collider.material.side = THREE.BackSide;
    //scene.add(collider);
    
    // Avatar dynamic settings
    const model = {
        "id": null,
        "model_path": model_path,
        "vrm": vrm, // the actual vrm object
        "hipsHeight": hipsHeight, // its original hips height, used for scaling loaded animation
        "objectContainer": object_container, // the actual 3d group containing the vrm scene, handle centered position/rotation/scaling
        "collider": collider,
        "expression": "none",
        "animation_mixer": new THREE.AnimationMixer(vrm.scene),
        "motion": {
            "name": "none",
            "animation": null,
            "timeout": null
        },
        "talkEnd": 0,
        "hitboxes": {},
        "motionQueue":[],
        "ttsQueue":[],
        "isPlayingTts": false,
        "currentTtsAudio": null,
        "targetExpressions": {},
        "currentExpressions": {},
    };

    // Hit boxes
    if (extension_settings.vrm.hitboxes) {
        for(const body_part in HITBOXES)
        {
            const bone = vrm.humanoid.getNormalizedBoneNode(HITBOXES[body_part]["bone"])
            if (bone !== null) {
                const position = new THREE.Vector3();
                position.setFromMatrixPosition(bone.matrixWorld);
                console.debug(DEBUG_PREFIX,"Creating hitbox for",body_part,"at",position);

                const size = HITBOXES[body_part]["size"];
                const offset = HITBOXES[body_part]["offset"];

                // Collider used to detect mouse click
                const boundingBox = new THREE.Box3(new THREE.Vector3(-size.x,-size.y,-size.z), new THREE.Vector3(size.x,size.y,size.z));
                const dimensions = new THREE.Vector3().subVectors( boundingBox.max, boundingBox.min );
                // make a BoxGeometry of the same size as Box3
                const boxGeo = new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z);
                // move new mesh center so it's aligned with the original object
                const matrix = new THREE.Matrix4().setPosition(dimensions.addVectors(boundingBox.min, boundingBox.max).multiplyScalar( 0.5 ));
                boxGeo.applyMatrix4(matrix);
                // make a mesh
                const collider = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({
                    visible: true,
                    side: THREE.BackSide,
                    wireframe: true,
                    color:HITBOXES[body_part]["color"]
                }));
                collider.name = body_part;
                if (vrm.meta?.metaVersion === '1')
                    collider.position.set(offset.x/hipsHeight,offset.y/hipsHeight,-offset.z/hipsHeight);
                else
                    collider.position.set(-offset.x/hipsHeight,offset.y/hipsHeight,offset.z/hipsHeight);
                // Create a offset container
                const offset_container = new THREE.Group(); // First container to scale/position center model
                offset_container.name = model_path+"_offsetContainer_hitbox_"+body_part;
                offset_container.visible = true;
                offset_container.add(collider);
                //scene.add(offset_container)

                //object_container.localToWorld(position);
                //position.add(new THREE.Vector3(offset.x,offset.y,offset.z));
                //collider.position.set(position.x,position.y,position.z);
                //scene.add(collider);

                model["hitboxes"][body_part] = {
                    "offsetContainer":offset_container,
                    "collider":collider
                }
            }
        }
    }

    //console.debug(DEBUG_PREFIX,vrm);

    // Cache model
    if (extension_settings.vrm.models_cache)
        models_cache[model_path] = model;

    await initModel(model);
    
    console.debug(DEBUG_PREFIX,"VRM fully loaded:",model_path);
    
    return model;
}

async function initModel(model) {
    const object_container = model["objectContainer"];
    const model_path = model["model_path"];

    object_container.scale.x = extension_settings.vrm.model_settings[model_path]['scale'];
    object_container.scale.y = extension_settings.vrm.model_settings[model_path]['scale'];
    object_container.scale.z = extension_settings.vrm.model_settings[model_path]['scale'];

    object_container.position.x = extension_settings.vrm.model_settings[model_path]['x'];
    object_container.position.y = extension_settings.vrm.model_settings[model_path]['y'];
    object_container.position.z = 0.0;

    object_container.rotation.x = extension_settings.vrm.model_settings[model_path]['rx'];
    object_container.rotation.y = extension_settings.vrm.model_settings[model_path]['ry'];
    object_container.rotation.z = 0.0;

    // Cache model animations
    if (extension_settings.vrm.animations_cache && animations_cache[model_path] === undefined) {
        animations_cache[model_path] = {};
        const animation_names = [extension_settings.vrm.model_settings[model_path]['animation_default']['motion']]
        for (const i in extension_settings.vrm.model_settings[model_path]['classify_mapping']) {
            animation_names.push(extension_settings.vrm.model_settings[model_path]['classify_mapping'][i]["motion"]);
        }

        let count = 0;
        for (const file of animations_files) {
            count++;
            for (const i of animation_names) {
                if(file.includes(i) && animations_cache[model_path][file] === undefined) {
                    console.debug(DEBUG_PREFIX,"Loading animation",file,count,"/",animations_files.length)
                    const clip = await loadAnimation(model["vrm"], model["hipsHeight"], file);
                    if (clip !== undefined)
                        animations_cache[model_path][file] = clip;
                }
            }
        }

        console.debug(DEBUG_PREFIX,"Cached animations:",animations_cache[model_path]);
    }
}

async function setExpression(character, value) {
    if (current_avatars[character] === undefined) return;

    const vrm = current_avatars[character]["vrm"];
    
    if (value === "none" || value === undefined) return; 

    // Reset base emotions to 0, EXCEPT mouth shapes (so TTS lip sync doesn't break)
    for(const expression in vrm.expressionManager.expressionMap) {
        if (!['aa','ee','ih','oh','ou','blink'].includes(expression)) {
            current_avatars[character].targetExpressions[expression] = 0.0;
        }
    }

    // Set the new target
    current_avatars[character].targetExpressions[value] = 1.0;
    current_avatars[character]["expression"] = value;
}

async function loadAnimation(vrm, hipsHeight, motion_file_path) {
    let clip;
    try {
        // Mixamo animation
        if (motion_file_path.endsWith(".fbx")) {
            //console.debug(DEBUG_PREFIX,"Loading fbx file",motion_file_path);

            // Load animation
            clip = await loadMixamoAnimation(motion_file_path, vrm, hipsHeight);
        }
        else
        if (motion_file_path.endsWith(".bvh")) {
            //console.debug(DEBUG_PREFIX,"Loading bvh file",motion_file_path);
            clip = await loadBVHAnimation(motion_file_path, vrm, hipsHeight);
        }
        else
        if (motion_file_path.endsWith(".vrma")) {
            // VRMA (VRM Animation) file
            const vrmaLoader = new VRMALoader();
            const result = await vrmaLoader.loadAsync(motion_file_path, vrm);
            clip = result ? result.clip : null;
            if (!clip) {
                toastr.error('Wrong animation file format:'+motion_file_path, DEBUG_PREFIX + ' cannot play animation', { timeOut: 10000, extendedTimeOut: 20000, preventDuplicates: true });
                return null;
            }
        }
        else {
            //console.debug(DEBUG_PREFIX,"UNSUPORTED animation file");
            toastr.error('Wrong animation file format:'+motion_file_path, DEBUG_PREFIX + ' cannot play animation', { timeOut: 10000, extendedTimeOut: 20000, preventDuplicates: true });
            return;
        }
    }
    catch(error) {
        //console.debug(DEBUG_PREFIX,"Something went wrong when loading animation file:",motion_file_path);
        toastr.error('Wrong animation file format:'+motion_file_path, DEBUG_PREFIX + ' cannot play animation', { timeOut: 10000, extendedTimeOut: 20000, preventDuplicates: true });
        return null;
    }
    return clip;
}

// Added returnToIdle parameter (defaults to true for hitboxes/commands)
async function setMotion(character, motion_file_path, loop=false, force=false, random=true, returnToIdle=true ) {
    if (current_avatars[character] === undefined) {
        console.debug(DEBUG_PREFIX,"WARNING requested setMotion of character without vrm loaded:",character);
        return;
    }
    const model_path = extension_settings.vrm.character_model_mapping[character];
    const vrm = current_avatars[character]["vrm"];
    const hipsHeight = current_avatars[character]["hipsHeight"];
    const mixer = current_avatars[character]["animation_mixer"];
    const current_motion_name = current_avatars[character]["motion"]["name"];
    const current_motion_animation= current_avatars[character]["motion"]["animation"];
    let clip = undefined;

    if (current_avatars[character]["motion"]["timeout"]) {
        clearTimeout(current_avatars[character]["motion"]["timeout"]);
        current_avatars[character]["motion"]["timeout"] = null;
    }

    if (motion_file_path == "none") {
        if (current_motion_animation !== null) {
            current_motion_animation.fadeOut(ANIMATION_FADE_TIME);
            current_motion_animation.terminated = true;
        }
        current_avatars[character]["motion"]["name"] = "none";
        current_avatars[character]["motion"]["animation"] = null;
        return;
    }

    const filename = motion_file_path.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
    let same_motion =[];
    for(const i of animations_files) {
        if (i.replace(/\.[^/.]+$/, "").replace(/\d+$/, "") == filename) {
            same_motion.push(i);
        }
    }
    
    if (same_motion.length > 0) {
        if (random) {
            motion_file_path = same_motion[Math.floor(Math.random() * same_motion.length)];
        } else {
            if (!motion_file_path.match(/\.(fbx|bvh|vrma)$/i)) {
                motion_file_path = same_motion[0];
            }
        }
    }

    if (current_motion_name != motion_file_path || loop || force) {

        if (animations_cache[model_path] !== undefined && animations_cache[model_path][motion_file_path] !== undefined) {
            clip = animations_cache[model_path][motion_file_path];
        } else {
            clip = await loadAnimation(vrm, hipsHeight, motion_file_path);
            if (clip === null) return;
            if (extension_settings.vrm.animations_cache) {
                if (!animations_cache[model_path]) animations_cache[model_path] = {};
                animations_cache[model_path][motion_file_path] = clip;
            }
        }

        const new_motion_animation = mixer.clipAction( clip );

        if ( current_motion_animation !== null ) {
            current_motion_animation.fadeOut( ANIMATION_FADE_TIME );
            current_motion_animation.terminated = true;
        }
        
        if (!loop) {
            new_motion_animation.clampWhenFinished = true;
            new_motion_animation.loop = THREE.LoopOnce;
        } else {
            new_motion_animation.clampWhenFinished = false;
            new_motion_animation.loop = THREE.LoopRepeat;
        }

        new_motion_animation
            .reset()
            .setEffectiveTimeScale( 1 )
            .setEffectiveWeight( 1 )
            .fadeIn( ANIMATION_FADE_TIME )
            .play();
        new_motion_animation.terminated = false;

        current_avatars[character]["motion"]["name"] = motion_file_path;
        current_avatars[character]["motion"]["animation"] = new_motion_animation;

        if (!loop && returnToIdle) {
            const timeoutId = setTimeout(() => {
                if (!new_motion_animation.terminated && current_avatars[character]["motion"]["timeout"] === timeoutId) {
                    if (current_avatars[character]["motionQueue"] && current_avatars[character]["motionQueue"].length > 0) {
                        const nextMotion = current_avatars[character]["motionQueue"].shift();
                        setMotion(character, nextMotion, false, true, true);
                    } else {
                        setMotion(character, extension_settings.vrm.model_settings[model_path]["animation_default"]["motion"], true);
                    }
                }
            }, clip.duration * 1000 - ANIMATION_FADE_TIME * 1000);
            
            current_avatars[character]["motion"]["timeout"] = timeoutId;
        }
    }
}

async function updateExpression(chat_id, skipMotion = false) {
    const message = getContext().chat[chat_id];
    const character = message.name;
    const model_path = extension_settings.vrm.character_model_mapping[character];

    console.debug(DEBUG_PREFIX,'received new message :', message.mes);

    if (message.is_user) return;
    if (model_path === undefined) {
        console.debug(DEBUG_PREFIX, 'No model assigned to', character);
        return;
    }

    const tags =[...message.mes.matchAll(/\[(.*?)\]/g)].map(m => m[1]);
    const timelineMotions =[];

    if (tags.length > 0) {
        const fuse = new Fuse(animations_files);
        for (const tag of tags) {
            const results = fuse.search(tag);
            const fileItem = results[0]?.item;
            if (fileItem) {
                timelineMotions.push(fileItem);
            }
        }
    }

    if (timelineMotions.length > 0) {
        if (!skipMotion) {
            console.debug(DEBUG_PREFIX, 'Playing timeline animations:', timelineMotions);
            playTimelineMotions(character, timelineMotions);
        }
        return; 
    }
}


// Blink
function blink(character, modelId) {
    if (current_avatars[character] === undefined || current_avatars[character]["id"] != modelId) {
        return;
    }
    
    const avatar = current_avatars[character];    
    if (avatar?.vrm?.expressionManager) {
        // Check for winking state and clear it
        const blinkLeftVal = avatar.vrm.expressionManager.getValue('blinkLeft') || 0;
        const blinkRightVal = avatar.vrm.expressionManager.getValue('blinkRight') || 0;
        if (blinkLeftVal > 0.1 || blinkRightVal > 0.1) {
            avatar.vrm.expressionManager.setValue('blinkLeft', 0);
            avatar.vrm.expressionManager.setValue('blinkRight', 0);
            avatar.winking = false;
            avatar.customWinking = false;
        }
    }

    // Close eyes smoothly by setting the target expression
    avatar.targetExpressions["blink"] = 1.0;

    // Hold eyes closed for a brief moment, then open smoothly
    var blinktimeout = Math.floor(Math.random() * 150) + 50; // 50-200ms
    setTimeout(() => {
        if (current_avatars[character] && current_avatars[character]["id"] == modelId) {
            avatar.targetExpressions["blink"] = 0.0;
        }
    }, blinktimeout);

    // Keep eyes open for a random duration before the next blink
    var rand = Math.round(Math.random() * 6000) + 2000; // 2 to 8 seconds
    setTimeout(function () {
        blink(character, modelId);
    }, rand);
}

// One run for each character
// Animate mouth if talkEnd is set to a future time
// Terminated when model is unset
// Overrided by tts lip sync option
async function textTalk(character, modelId) {
    const mouth_open_speed = 1.5;
    // Model still here
    while (current_avatars[character] !== undefined && current_avatars[character]["id"] == modelId) {
        //console.debug(DEBUG_PREFIX,"text talk loop:",character,modelId)
        
        // Overrided by lip sync option
        if (!extension_settings.vrm.tts_lips_sync) {
            const vrm = current_avatars[character]["vrm"]
            const talkEnd = current_avatars[character]["talkEnd"]
            let mouth_y = 0.0;
            if (talkEnd > Date.now()) {
                mouth_y = (Math.sin((talkEnd - Date.now())) + 1) / 2;
                vrm.expressionManager.setValue("aa",mouth_y);
            }
            else {
                vrm.expressionManager.setValue("aa",0.0);
            }
        }
        await delay(100 / mouth_open_speed);
    }

    console.debug(DEBUG_PREFIX,"Stopping text talk loop model is no more loaded:",character,modelId);
}

// Add text duration to current_avatars[character]["talkEnd"]
// Overrided by tts lip sync option
async function talk(chat_id) {
    // TTS lip sync overide
    if (extension_settings.vrm.tts_lips_sync)
        return;

    // No model for user or system
    if (getContext().chat[chat_id].is_user || getContext().chat[chat_id].is_system)
        return;

    const message = getContext().chat[chat_id]
    const text = message.mes;
    const character = message.name;

    console.debug(DEBUG_PREFIX,"Playing mouth animation for",character," message:",text);

    // No model loaded for character
    if(current_avatars[character] === undefined) {
        console.debug(DEBUG_PREFIX,"No model loaded, cannot animate talk")
        return;
    }

    current_avatars[character]["talkEnd"] = Date.now() + text.length * 50;
}

// handle window resizes
window.addEventListener( 'resize', onWindowResize, false );

function onWindowResize(){
    if (camera !== undefined && renderer !== undefined) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();

        renderer.setSize( window.innerWidth, window.innerHeight );
    }
}

// Update a character model to fit the saved settings
async function updateModel(character) {
    if (current_avatars[character] !== undefined) {
        const object_container = current_avatars[character]["objectContainer"];
        const model_path = extension_settings.vrm.character_model_mapping[character];

        object_container.scale.x = extension_settings.vrm.model_settings[model_path]['scale'];
        object_container.scale.y = extension_settings.vrm.model_settings[model_path]['scale'];
        object_container.scale.z = extension_settings.vrm.model_settings[model_path]['scale'];

        object_container.position.x = extension_settings.vrm.model_settings[model_path]['x'];
        object_container.position.y = extension_settings.vrm.model_settings[model_path]['y'];
        object_container.position.z = extension_settings.vrm.model_settings[model_path]['z']; //0.0; // In case somehow it get away from 0

        object_container.rotation.x = extension_settings.vrm.model_settings[model_path]['rx'];
        object_container.rotation.y = extension_settings.vrm.model_settings[model_path]['ry'];
        object_container.rotation.z = extension_settings.vrm.model_settings[model_path]['rz']; //0.0; // In case somehow it get away from 0

        console.debug(DEBUG_PREFIX,"Updated model:",character)
        console.debug(DEBUG_PREFIX,"Scale:",object_container.scale)
        console.debug(DEBUG_PREFIX,"Position:",object_container.position)
        console.debug(DEBUG_PREFIX,"Rotation:",object_container.rotation)
    }
}

// Currently loaded character VRM accessor
function getVRM(character) {
    if (current_avatars[character] === undefined)
        return undefined;
    return current_avatars[character]["vrm"];
}

function clearModelCache() {
    models_cache = {};
    console.debug(DEBUG_PREFIX,"Cleared model cache");
}

function clearAnimationCache() {
    animations_cache = {};
    console.debug(DEBUG_PREFIX,"Cleared animation cache");
}

// Perform audio lip sync
// Overried text mouth movement
async function audioTalk(blob, character) {
    // Option disable
    if (!extension_settings.vrm.tts_lips_sync)
        return;
        /*return response;

    console.debug(DEBUG_PREFIX,"Received lipsync",response, character);
    let responseCopy;
    try {
        responseCopy = response.clone();
    } catch(error) {
        console.debug(DEBUG_PREFIX,"Wrong response format received abort lip sync");
        return response;
    }*/
    tts_lips_sync_job_id++;
    const job_id = tts_lips_sync_job_id;
    console.debug(DEBUG_PREFIX,"Received lipsync",blob, character,job_id);

    const audioContext = new(window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.smoothingTimeConstant = 0.5;
    analyser.fftSize = 1024;

    //const blob = await responseCopy.blob();
    const arrayBuffer = await blob.arrayBuffer();

    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);

    const javascriptNode = audioContext.createScriptProcessor(256, 1, 1);
    analyser.connect(javascriptNode);
    javascriptNode.connect(audioContext.destination);
    const mouththreshold = 10;
    const mouthboost = 10;

    let lastUpdate = 0;
    const LIPS_SYNC_DELAY = 66;

    function endTalk() {
        source.stop(0);
        source.disconnect();
        analyser.disconnect();
        javascriptNode.disconnect();
        if (current_avatars[character] !== undefined)
            current_avatars[character]["vrm"].expressionManager.setValue("aa", 0);

        audio.removeEventListener("ended", endTalk);
        //javascriptNode.removeEventListener("onaudioprocess", onAudioProcess);
    }

    var audio = document.getElementById("tts_audio");
    function startTalk() {
        source.start(0);
        audio.removeEventListener("onplay", startTalk);
        //javascriptNode.removeEventListener("onaudioprocess", onAudioProcess);
    }
    audio.onplay = startTalk;
    audio.onended = endTalk;

    function onAudioProcess() {
        if(job_id != tts_lips_sync_job_id || audio.paused) {
            console.debug(DEBUG_PREFIX,"TTS lip sync job",job_id,"terminated")
            endTalk();
            return;
        }

        var array = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(array);
        var values = 0;

        var length = array.length;
        for (var i = 0; i < length; i++) {
            values += array[i];
        }

        // audio in expressed as one number
        var average = values / length;
        var inputvolume = average * (audioContext.sampleRate/48000); // Normalize the treshold

        var voweldamp = 53;
        var vowelmin = 12;
        if(lastUpdate < (Date.now() - LIPS_SYNC_DELAY)) {
            if (current_avatars[character] !== undefined) {

                if (inputvolume > (mouththreshold * 2)) {
                    const new_value = ((average - vowelmin) / voweldamp) * (mouthboost/10);
                    current_avatars[character]["vrm"].expressionManager.setValue("aa", new_value);
                }
                else {
                    current_avatars[character]["vrm"].expressionManager.setValue("aa", 0);
                }
            }
            lastUpdate = Date.now();
        }
    }

    javascriptNode.onaudioprocess = onAudioProcess;
    // TODO: restaure expression weight ?
}

window['vrmLipSync'] = audioTalk;

// color: any valid color format
// intensity: percent 0-100
function setLight(color,intensity) {

    light.color = new THREE.Color(color);
    light.intensity = intensity/100;
}

function setBackground(scenePath, scale, position, rotation) {

    if (background) {
        scene.remove(scene.getObjectByName(background.name));
    }

    if (scenePath.endsWith(".fbx")) {
        const fbxLoader = new FBXLoader()
        fbxLoader.load(
            scenePath,
        (object) => {
            // object.traverse(function (child) {
            //     if ((child as THREE.Mesh).isMesh) {
            //         // (child as THREE.Mesh).material = material
            //         if ((child as THREE.Mesh).material) {
            //             ((child as THREE.Mesh).material as THREE.MeshBasicMaterial).transparent = false
            //         }
            //     }
            // })
            // object.scale.set(.01, .01, .01)
            background = object;
            background.scale.set(scale, scale, scale);
            background.position.set(position.x,position.y,position.z);
            background.rotation.set(rotation.x,rotation.y,rotation.z);
            background.name = "background";
            scene.add(background);
        },
        (xhr) => {
            console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
        },
        (error) => {
            console.log(error)
        }
        )
    }

    if (scenePath.endsWith(".gltf")) {
        const loader = new GLTFLoader();

        loader.load( scenePath, function ( gltf ) {

            background = gltf.scene;
            background.scale.set(scale, scale, scale);
            background.position.set(position.x,position.y,position.z);
            background.rotation.set(rotation.x,rotation.y,rotation.z);
            scene.add(background);

        }, undefined, function ( error ) {

            console.error( error );

        } );
    }
}

async function playTimelineMotions(character, motionsArray) {
    if (!current_avatars[character]) return;
    if (!motionsArray || motionsArray.length === 0) return;

    // Set the queue and trigger the first animation immediately
    current_avatars[character]["motionQueue"] = motionsArray;
    const firstMotion = current_avatars[character]["motionQueue"].shift();
    
    // Play it (loop=false, force=true, random=true)
    await setMotion(character, firstMotion, false, true, true);
}

// Helper to convert Base64 directly into a playable Blob
function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], {type: mimeType});
}

// Immediately stops current TTS and resets the mouth
function stopTTS(character) {
    const avatar = current_avatars[character];
    if (!avatar) return;

    avatar.ttsQueue =[]; 
    if (avatar.currentTtsAudio) {
        avatar.currentTtsAudio.pause();
        avatar.currentTtsAudio.currentTime = 0;
        avatar.currentTtsAudio = null;
    }
    
    avatar.isPlayingTts = false;
    
    const shapes =['aa', 'ee', 'ih', 'oh', 'ou'];
    shapes.forEach(shape => {
        if (avatar.vrm && avatar.vrm.expressionManager) {
            avatar.vrm.expressionManager.setValue(shape, 0);
        }
    });

    const model_path = extension_settings.vrm.character_model_mapping[character];
    if (model_path) {
        const defaultMot = extension_settings.vrm.model_settings[model_path]['animation_default']['motion'];
        
        let currentMotionGroup = avatar.motion.name;
        if (currentMotionGroup && currentMotionGroup !== "none") {
            currentMotionGroup = currentMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
        }
        let defaultMotionGroup = defaultMot;
        if (defaultMotionGroup && defaultMotionGroup !== "none") {
            defaultMotionGroup = defaultMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
        }

        if (currentMotionGroup !== defaultMotionGroup) setMotion(character, defaultMot, true, false, false);
    }
}


// Replaces the old audioTalk with a direct MediaElement binder
function attachVolumeLipSync(audio, character) {
    if (!extension_settings.vrm.tts_lips_sync) return;

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.smoothingTimeConstant = 0.5;
    analyser.fftSize = 1024;

    // Bind directly to our queued audio object
    const source = audioContext.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    const javascriptNode = audioContext.createScriptProcessor(256, 1, 1);
    analyser.connect(javascriptNode);
    javascriptNode.connect(audioContext.destination);

    const mouththreshold = 10;
    const mouthboost = 10;
    let lastUpdate = 0;
    const LIPS_SYNC_DELAY = 66;

    javascriptNode.onaudioprocess = function() {
        if (audio.paused || audio.ended) return;

        const array = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(array);
        let values = 0;
        for (let i = 0; i < array.length; i++) {
            values += array[i];
        }
        const average = values / array.length;
        const inputvolume = average * (audioContext.sampleRate / 48000);

        const voweldamp = 53;
        const vowelmin = 12;

        if (Date.now() - lastUpdate > LIPS_SYNC_DELAY) {
            const avatar = current_avatars[character];
            if (avatar && avatar.vrm) {

                if (inputvolume > (mouththreshold * 2)) {
                    const new_value = ((average - vowelmin) / voweldamp) * (mouthboost / 10);
                    avatar.vrm.expressionManager.setValue("aa", new_value);
                } else {
                    avatar.vrm.expressionManager.setValue("aa", 0);
                }
            }
            lastUpdate = Date.now();
        }
    };

    // Cleanup when audio ends
    audio.addEventListener("ended", () => {
        source.disconnect();
        analyser.disconnect();
        javascriptNode.disconnect();
        if (audioContext.state !== 'closed') {
            audioContext.close();
        }
    }, { once: true });
}

async function processAndQueueTTS(character, text, clearQueue = false) {
    const avatar = current_avatars[character];
    if (!avatar) return;

    if (clearQueue) stopTTS(character);

    const sentenceObjects = extractSentencesWithContext(text);
    if (!sentenceObjects || sentenceObjects.length === 0) return;
    
    let voiceId = extension_settings.vrm.inworld_default_voice_id || "Dennis";
    if (extension_settings.vrm.voiceMap && extension_settings.vrm.voiceMap[character]) {
        voiceId = extension_settings.vrm.voiceMap[character];
    }

    const temperature = extension_settings.vrm.inworld_temperature ?? 1.1;
    const speed = extension_settings.vrm.inworld_speed ?? 1.0;

    let availableExpressions =[];
    if (avatar.vrm && avatar.vrm.expressionManager) {
        availableExpressions = Object.keys(avatar.vrm.expressionManager.expressionMap).filter(
            e => !avatar.vrm.expressionManager.blinkExpressionNames.includes(e) && 
                 !avatar.vrm.expressionManager.mouthExpressionNames.includes(e) && 
                 !avatar.vrm.expressionManager.lookAtExpressionNames.includes(e)
        );
    }
    const availableMotions = animations_groups ||[];

    for (const item of sentenceObjects) {
        const { sentence, textBefore, textAfter } = item;

        const [ttsData, tags] = await Promise.all([
            fetchInworldTTS(sentence, voiceId, temperature, speed),
            fetchSmallLLMTag(sentence, textBefore, textAfter, availableExpressions, availableMotions)
        ]);

        if (ttsData && ttsData.audioContent) {
            avatar.ttsQueue.push({
                audioBase64: ttsData.audioContent,
                expression: tags.expression,
                motion: tags.motion
            });

            playNextInQueue(character);
        }
    }
}

async function playNextInQueue(character) {
    const avatar = current_avatars[character];
    if (!avatar) return;

    if (avatar.ttsQueue.length === 0) {
        if (!avatar.isPlayingTts) {
            const model_path = extension_settings.vrm.character_model_mapping[character];
            if (model_path) {
                const defaultMot = extension_settings.vrm.model_settings[model_path]['animation_default']['motion'];
                
                let currentMotionGroup = avatar.motion.name;
                if (currentMotionGroup && currentMotionGroup !== "none") {
                    currentMotionGroup = currentMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
                }
                let defaultMotionGroup = defaultMot;
                if (defaultMotionGroup && defaultMotionGroup !== "none") {
                    defaultMotionGroup = defaultMotionGroup.replace(/\.[^/.]+$/, "").replace(/\d+$/, "");
                }

                if (currentMotionGroup !== defaultMotionGroup) setMotion(character, defaultMot, true, false, false);
            }
        }
        return;
    }

    if (avatar.isPlayingTts) return;

    avatar.isPlayingTts = true;
    const item = avatar.ttsQueue.shift();

    const blob = base64ToBlob(item.audioBase64, 'audio/mp3');
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.volume = 1.0; 
    
    avatar.currentTtsAudio = audio;

    if (item.expression && item.expression !== "none") {
        setExpression(character, item.expression);
    }
    if (item.motion && item.motion !== "none") {
        setMotion(character, item.motion, false, true, true, false); 
    }

    await new Promise(resolve => setTimeout(resolve, 800));

    if (!avatar.isPlayingTts || avatar.currentTtsAudio !== audio) {
        URL.revokeObjectURL(url);
        return; 
    }

    attachVolumeLipSync(audio, character);

    audio.onended = () => {
        URL.revokeObjectURL(url); 
        avatar.isPlayingTts = false;
        avatar.currentTtsAudio = null;['aa', 'ee', 'ih', 'oh', 'ou'].forEach(shape => {
            if (avatar.vrm && avatar.vrm.expressionManager) {
                avatar.vrm.expressionManager.setValue(shape, 0);
            }
        });
        
        playNextInQueue(character);
    };

    audio.play().catch(e => {
        console.error(DEBUG_PREFIX, "Audio playback blocked by browser:", e);
        URL.revokeObjectURL(url);
        avatar.isPlayingTts = false;
        playNextInQueue(character);
    });
}

function blendExpressions(character, weights) {
    if (!current_avatars[character]) return;
    const vrm = current_avatars[character].vrm;

    // Zero out base emotions first
    const baseEmotions =['happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral'];
    for (const expr of baseEmotions) {
        current_avatars[character].targetExpressions[expr] = 0.0;
    }

    // Apply the new mixed weights from Groq
    for (const [expr, val] of Object.entries(weights)) {
        if (vrm.expressionManager.expressionMap[expr] !== undefined || baseEmotions.includes(expr)) {
            current_avatars[character].targetExpressions[expr] = val;
        }
    }
}