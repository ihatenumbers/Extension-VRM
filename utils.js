import { getContext, extension_settings, getApiUrl, doExtrasFetch, modules } from '../../../extensions.js';
import { getRequestHeaders, saveSettings, saveSettingsDebounced, substituteParams, eventSource, event_types, generateQuietPrompt } from '../../../../script.js';
import { isJsonSchemaSupported } from '../../../textgen-settings.js';
import {
    trimToEndSentence,
    trimToStartSentence,
    onlyUnique } from '../../../utils.js';
import {
    DEBUG_PREFIX,
    DEFAULT_EXPRESSION_MAPPING,
    DEFAULT_MOTION_MAPPING,
    FALLBACK_EXPRESSION,
    CLASSIFY_EXPRESSIONS
} from './constants.js';
export {
    delay,
    currentChatMembers,
    loadAnimationUi,
    getExpressionLabel,
    extractDialogue,
    chunkText,
    extractSentencesWithContext,
    fetchInworldTTS,
    fetchSmallLLMTag
};

const delay = ms => new Promise(res => setTimeout(res, ms));

// Expression extension code
const EXPRESSION_API = {
    local: 0,
    extras: 1,
    llm: 2,
};
let expressionsList = null;
let inApiCall = false;

function currentChatMembers() {
    const context = getContext();
    const group_id = context.groupId;
    let chat_members = [context.name2];

    if (group_id !== null) {
        chat_members = [];
        for(const i of context.groups) {
            if (i.id == context.groupId) {
                for(const j of i.members) {
                    let char_name = j.replace(/\.[^/.]+$/, '');
                    if (char_name.includes('default_'))
                        char_name = char_name.substring('default_'.length);

                    chat_members.push(char_name);
                }
            }
        }
    }

    chat_members.sort();

    return chat_members;
}

function loadAnimationUi(type, use_default_settings, model_expressions, model_motions, expression_select_id, motion_select_id, expression_select_value, motion_select_value, default_settings=false) {
    $(`#${expression_select_id}`)
        .find('option')
        .remove()
        .end()
        .append('<option value="none">Select expression</option>');

    $(`#${motion_select_id}`)
        .find('option')
        .remove()
        .end()
        .append('<option value="none">Select motion</option>');

    for (const expression of model_expressions) {
        $(`#${expression_select_id}`).append(new Option(expression, expression));
    }

    for (const motion of model_motions) {
        const name = motion.substring(motion.lastIndexOf('/')+1).replace(".fbx","").replace(".bvh","");
        $(`#${motion_select_id}`).append(new Option(name, motion));
    }

    
    $(`#${expression_select_id}`).val(expression_select_value);
    $(`#${motion_select_id}`).val(motion_select_value);

    if (use_default_settings) {
        if (model_expressions.includes(DEFAULT_EXPRESSION_MAPPING[type])) {
            $(`#${expression_select_id}`).val(DEFAULT_EXPRESSION_MAPPING[type]);
        }
        if (model_motions.includes(DEFAULT_MOTION_MAPPING[type])) {
            $(`#${motion_select_id}`).val(DEFAULT_MOTION_MAPPING[type]);
        }
    }
}

// Copied from expression extension
async function getExpressionsList() {
    // Return cached list if available
    if (Array.isArray(expressionsList)) {
        return [...expressionsList, ...extension_settings.expressions.custom].filter(onlyUnique);
    }

    /**
     * Returns the list of expressions from the API or fallback in offline mode.
     * @returns {Promise<string[]>}
     */
    async function resolveExpressionsList() {
        // See if we can retrieve a specific expression list from the API
        try {
            // Check Extras api first, if enabled and that module active
            if (extension_settings.expressions.api == EXPRESSION_API.extras && modules.includes('classify')) {
                const url = new URL(getApiUrl());
                url.pathname = '/api/classify/labels';

                const apiResult = await doExtrasFetch(url, {
                    method: 'GET',
                    headers: { 'Bypass-Tunnel-Reminder': 'bypass' },
                });

                if (apiResult.ok) {

                    const data = await apiResult.json();
                    expressionsList = data.labels;
                    return expressionsList;
                }
            }

            // If running the local classify model (not using the LLM), we ask that one
            if (extension_settings.expressions.api == EXPRESSION_API.local) {
                const apiResult = await fetch('/api/extra/classify/labels', {
                    method: 'POST',
                    headers: getRequestHeaders(),
                });

                if (apiResult.ok) {
                    const data = await apiResult.json();
                    expressionsList = data.labels;
                    return expressionsList;
                }
            }
        } catch (error) {
            console.log(error);
        }

        // If there was no specific list, or an error, just return the default expressions
        return CLASSIFY_EXPRESSIONS;
    }

    const result = await resolveExpressionsList();
    return [...result, ...extension_settings.expressions.custom].filter(onlyUnique);
}

/**
 * Gets the classification prompt for the LLM API.
 * @param {string[]} labels A list of labels to search for.
 * @returns {Promise<string>} Prompt for the LLM API.
 */
async function getLlmPrompt(labels) {
    if (isJsonSchemaSupported()) {
        return '';
    }

    const labelsString = labels.map(x => `"${x}"`).join(', ');
    const prompt = substituteParams(String(extension_settings.expressions.llmPrompt))
        .replace(/{{labels}}/gi, labelsString);
    return prompt;
}

function onTextGenSettingsReady(args) {
    // Only call if inside an API call
    if (inApiCall && extension_settings.expressions.api === EXPRESSION_API.llm && isJsonSchemaSupported()) {
        const emotions = DEFAULT_EXPRESSIONS.filter((e) => e != 'talkinghead');
        Object.assign(args, {
            top_k: 1,
            stop: [],
            stopping_strings: [],
            custom_token_bans: [],
            json_schema: {
                $schema: 'http://json-schema.org/draft-04/schema#',
                type: 'object',
                properties: {
                    emotion: {
                        type: 'string',
                        enum: emotions,
                    },
                },
                required: [
                    'emotion',
                ],
            },
        });
    }
}

async function getExpressionLabel(text) {
    
    // Return if text is undefined, saving a costly fetch request
    //if ((!modules.includes('classify') && !extension_settings.expressions.local) || !text) {
    if ((!modules.includes('classify') && extension_settings.expressions.api == EXPRESSION_API.extras) || !text) {
        return FALLBACK_EXPRESSION;
    }

    if (extension_settings.expressions.translate && typeof window['translate'] === 'function') {
        text = await window['translate'](text, 'en');
    }

    text = sampleClassifyText(text);

    try {
        switch (extension_settings.expressions.api) {
            // Local BERT pipeline
            case EXPRESSION_API.local: {
                const localResult = await fetch('/api/extra/classify', {
                    method: 'POST',
                    headers: getRequestHeaders(),
                    body: JSON.stringify({ text: text }),
                });

                if (localResult.ok) {
                    const data = await localResult.json();
                    return data.classification[0].label;
                }
            } break;
            // Using LLM
            case EXPRESSION_API.llm: {
                const expressionsList = await getExpressionsList();
                const prompt = await getLlmPrompt(expressionsList);
                eventSource.once(event_types.TEXT_COMPLETION_SETTINGS_READY, onTextGenSettingsReady);
                const emotionResponse = await generateQuietPrompt(prompt, false, false);
                return parseLlmResponse(emotionResponse, expressionsList);
            }
            // Extras
            default: {
                const url = new URL(getApiUrl());
                url.pathname = '/api/classify';

                const extrasResult = await doExtrasFetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Bypass-Tunnel-Reminder': 'bypass',
                    },
                    body: JSON.stringify({ text: text }),
                });

                if (extrasResult.ok) {
                    const data = await extrasResult.json();
                    return data.classification[0].label;
                }
            } break;
        }
    } catch (error) {
        toastr.info('Could not classify expression. Check the console or your backend for more information.');
        console.error(error);
        return FALLBACK_EXPRESSION;
    }
}

/**
 * Extracts dialogue sentences and maps them to their surrounding narrative context.
 */
function extractSentencesWithContext(text) {
    const sentences = [];
    const hasQuotes = /"([^"]+)"/.test(text);
    const hasAsterisks = /\*([^*]+)\*/.test(text);

    if (hasQuotes) {
        const regex = /"([^"]+)"/g;
        let match;
        let lastIndex = 0;
        const dialogues =[];

        // Extract dialogue and the narrative immediately before it
        while ((match = regex.exec(text)) !== null) {
            const narrativeBefore = text.substring(lastIndex, match.index).trim();
            const dialogueText = match[1].trim();
            dialogues.push({ dialogue: dialogueText, narrativeBefore: narrativeBefore, narrativeAfter: "" });
            lastIndex = regex.lastIndex;
        }
        
        // The remaining text is the narrative after the final dialogue
        const trailingNarrative = text.substring(lastIndex).trim();
        
        // Link the narrativeAfter for each dialogue chunk
        for (let i = 0; i < dialogues.length; i++) {
            if (i < dialogues.length - 1) {
                dialogues[i].narrativeAfter = dialogues[i+1].narrativeBefore;
            } else {
                dialogues[i].narrativeAfter = trailingNarrative;
            }
        }
        
        // Chunk dialogues into sentences and attach the mapped context
        for (const d of dialogues) {
            const chunks = chunkText(d.dialogue);
            for (const chunk of chunks) {
                sentences.push({ sentence: chunk, textBefore: d.narrativeBefore, textAfter: d.narrativeAfter });
            }
        }
    } else if (hasAsterisks) {
        // Fallback for roleplayers who use *asterisks* for narrative instead of quotes
        const regex = /\*([^*]+)\*/g;
        let match;
        let lastIndex = 0;
        let narrativeBefore = "";
        let dialogues =[];

        while ((match = regex.exec(text)) !== null) {
            const dialogueText = text.substring(lastIndex, match.index).trim();
            const narrativeText = match[1].trim();
            
            if (dialogueText) {
                dialogues.push({ dialogue: dialogueText, narrativeBefore: narrativeBefore, narrativeAfter: narrativeText });
            }
            narrativeBefore = narrativeText;
            lastIndex = regex.lastIndex;
        }
        const trailingDialogue = text.substring(lastIndex).trim();
        if (trailingDialogue) {
            dialogues.push({ dialogue: trailingDialogue, narrativeBefore: narrativeBefore, narrativeAfter: "" });
        }

        for (const d of dialogues) {
            const chunks = chunkText(d.dialogue);
            for (const chunk of chunks) {
                sentences.push({ sentence: chunk, textBefore: d.narrativeBefore, textAfter: d.narrativeAfter });
            }
        }
    } else {
        // No markup, treat the whole block as dialogue
        const chunks = chunkText(text);
        for (const chunk of chunks) {
            sentences.push({ sentence: chunk, textBefore: "", textAfter: "" });
        }
    }
    return sentences;
}

/**
 * Extracts only the text inside "quotes". Fallback to removing *asterisks* if no quotes exist.
 */
function extractDialogue(text) {
    const matches = text.match(/"([^"]+)"/g);
    if (matches) {
        return matches.map(m => m.replace(/"/g, '')).join(' ');
    }
    // Fallback: remove roleplay asterisks
    return text.replace(/\*[^*]+\*/g, '').trim(); 
}

function chunkText(text) {
    if (!text) return[];
    // Split by punctuation (., !, ?) keeping the punctuation attached.
    return text.match(/[^.!?]+[.!?]+|\s*[^.!?]+$/g)?.map(s => s.trim()).filter(s => s.length > 0) || [text];
}

const ttsCache = new Map();

/**
 * Fetches the audio and timestamps from Inworld.
 */
async function fetchInworldTTS(text, voiceId, temperature, speed) {
    const apiKey = extension_settings.vrm.inworld_api_key || ""; 
    
    if (!apiKey) {
        console.warn(DEBUG_PREFIX, "Inworld API Key is missing.");
        return null;
    }

    // Cache Check: Prevents redundant API calls for repeated dialogue
    const cacheKey = `${text}|${voiceId}|${temperature}|${speed}`;
    if (ttsCache.has(cacheKey)) {
        console.debug(DEBUG_PREFIX, "Using cached Inworld TTS audio.");
        return ttsCache.get(cacheKey);
    }

    try {
        const response = await fetch("https://api.inworld.ai/tts/v1/voice", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Basic ${apiKey}`
            },
            body: JSON.stringify({
                text: text,
                voiceId: voiceId,
                modelId: "inworld-tts-1.5-max",
                temperature: temperature,
                audioConfig: { 
                    audioEncoding: "MP3", 
                    sampleRateHertz: 44100,
                    speakingRate: speed
                },
                timestampType: "WORD"
            })
        });

        if (!response.ok) throw new Error(`Inworld API Error: ${response.status}`);
        const data = await response.json();
        
        // Save to cache
        ttsCache.set(cacheKey, data);
        
        return data;
    } catch (error) {
        console.error(DEBUG_PREFIX, "Inworld TTS fetch failed:", error);
        return null;
    }
}

const llmTagCache = new Map();

/**
 * Uses Groq API to fetch the best expression and animation based on context.
 */
async function fetchSmallLLMTag(sentence, textBefore, textAfter, availableExpressions, availableMotions) {
    const apiKey = extension_settings.vrm.groq_api_key || "";
    if (!apiKey) return { expression: null, motion: null };

    // Cache Check: Prevents redundant LLM calls
    const cacheKey = `${sentence}|${textBefore}|${textAfter}`;
    if (llmTagCache.has(cacheKey)) {
        console.debug(DEBUG_PREFIX, "Using cached LLM tag:", llmTagCache.get(cacheKey));
        return llmTagCache.get(cacheKey);
    }

    const motionMap = {};
    const shortMotions =[];
    
    for (const fullPath of availableMotions) {
        // Extract just the filename (e.g., "/assets/vrm/animation/anger" -> "anger")
        const shortName = fullPath.substring(fullPath.lastIndexOf('/') + 1);
        motionMap[shortName] = fullPath;
        shortMotions.push(shortName);
    }

    const systemPrompt = `You are an animation director for a 3D avatar.
Choose ONE expression and ONE animation that best match the "Current Sentence".

Available Expressions: ${availableExpressions.join(', ')}
Available Animations: ${shortMotions.join(', ')}

Reply EXACTLY with this format and nothing else:
[expression:NAME] [animation:NAME]

If nothing fits perfectly, use [expression:neutral] [animation:neutral].`;

    let userPrompt = `Current Sentence: "${sentence}"`;
    if (textBefore) userPrompt = `Context Before: "${textBefore}"\n` + userPrompt;
    if (textAfter) userPrompt += `\nContext After: "${textAfter}"`;

    try {
        const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: "moonshotai/kimi-k2-instruct-0905",
                messages:[
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt }
                ],
                temperature: 0.3,
                max_tokens: 30
            })
        });

        if (!response.ok) throw new Error(`Groq API Error: ${response.status}`);
        const data = await response.json();
        const content = data.choices[0].message.content;
        
        // Parse the tags: [expression:cheekpuff] [animation:anger]
        let expressionMatch = content.match(/\[expression:(.*?)\]/i);
        let motionMatch = content.match(/\[animation:(.*?)\]/i);

        let result = {
            expression: expressionMatch ? expressionMatch[1].trim() : null,
            motion: motionMatch ? motionMatch[1].trim() : null
        };

        if (result.expression && result.expression !== "none" && !availableExpressions.includes(result.expression)) {
            result.expression = null;
        }

        // --- NEW: VALIDATE SHORT NAME AND MAP BACK TO FULL PATH ---
        if (result.motion && result.motion !== "none") {
            if (shortMotions.includes(result.motion)) {
                // LLM said "anger", we map it to "/assets/vrm/animation/anger"
                result.motion = motionMap[result.motion]; 
            } else {
                result.motion = null; // Hallucinated animation, skip it
            }
        }

        console.debug(DEBUG_PREFIX, "System prompt: ", systemPrompt, "| User prompt: ", userPrompt, "| Groq Output:", content, "| Mapped Tags:", result);

        llmTagCache.set(cacheKey, result);
        return result;

    } catch (error) {
        console.error(DEBUG_PREFIX, "Groq LLM fetch failed:", error);
        return { expression: null, motion: null };
    }
}

/**
 * Parses the emotion response from the LLM API.
 * @param {string} emotionResponse The response from the LLM API.
 * @param {string[]} labels A list of labels to search for.
 * @returns {string} The parsed emotion or the fallback expression.
 */
function parseLlmResponse(emotionResponse, labels) {
    const fallbackExpression = FALLBACK_EXPRESSION;

    try {
        const parsedEmotion = JSON.parse(emotionResponse);
        return parsedEmotion?.emotion ?? fallbackExpression;
    } catch {
        const fuse = new Fuse([emotionResponse]);
        for (const label of labels) {
            const result = fuse.search(label);
            if (result.length > 0) {
                return label;
            }
        }
    }

    throw new Error('Could not parse emotion response ' + emotionResponse);
}

/**
 * Processes the classification text to reduce the amount of text sent to the API.
 * Quotes and asterisks are to be removed. If the text is less than 300 characters, it is returned as is.
 * If the text is more than 300 characters, the first and last 150 characters are returned.
 * The result is trimmed to the end of sentence.
 * @param {string} text The text to process.
 * @returns {string}
 */
function sampleClassifyText(text) {
    if (!text) {
        return text;
    }

    // Remove asterisks and quotes
    let result = text.replace(/[\*\"]/g, '');

    const SAMPLE_THRESHOLD = 300;
    const HALF_SAMPLE_THRESHOLD = SAMPLE_THRESHOLD / 2;

    if (text.length < SAMPLE_THRESHOLD) {
        result = trimToEndSentence(result);
    } else {
        result = trimToEndSentence(result.slice(0, HALF_SAMPLE_THRESHOLD)) + ' ' + trimToStartSentence(result.slice(-HALF_SAMPLE_THRESHOLD));
    }

    return result.trim();
}