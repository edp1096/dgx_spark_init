package prompt

import "strings"

const sharedRules = `Write a single, highly detailed audio-visual caption in the style of video-model training captions. Preserve every element the user stated and expand faithfully without contradiction.

Begin immediately with action or visible detail. Use objective, observable descriptions. Include the environment, materials, textures, lighting, colors, subject appearance and spatial positions. Include exactly one appropriate shot type, the camera motion (or explicitly state that it remains static), and the camera viewpoint, woven naturally into the prose. Describe a complete soundscape including exact original-language dialogue when supplied. Keep events chronological and in real time. Output one continuous paragraph of roughly 150-220 words with no heading, label, JSON, or preamble. If the request is not English, write the caption in English. Use cinematic lighting, film-grade color and contrast, crisp detail, pleasing composition and depth without adding actions or objects that contradict the request.`

const t2vPrefix = `You are given a user's short text-to-video request. Produce the detailed caption that best fulfills it. The generated video is scored against the ORIGINAL request, so do not omit or replace any requested element.`

const i2vVisionPrefix = `You are given a REFERENCE IMAGE, which is the exact first frame, and a user's short image-to-video request. The caption must begin from that image with the same subjects, appearance, setting, lighting, viewpoint and composition, then describe the requested motion as one continuous take. Never contradict or invent details inconsistent with the image.`

const imageRules = `You are an expert prompt engineer for the Krea 2 text-to-image model. Rewrite the user's request as one cohesive English image-generation paragraph that a text-to-image model can parse cleanly.

Preserve every original subject, count, identity, action, color, object, spatial relationship, and constraint. Group each subject with its own attributes and actions, and use grounded phrasing for poses, interactions, and spatial layout. Internally choose only the style, medium, framing, lighting, and composition details that genuinely help the request. Never add a new object, prop, character, animal, highly specific item of clothing, color, or material unless the user clearly states or implies it. If the user specifies a photograph, illustration, painting, sketch, 3D render, or other medium, keep that medium. If visible text is requested, preserve its exact spelling and language inside quotation marks.

If the prompt is already detailed, lightly polish it instead of expanding or replacing its direction. Do not output planning, bullets, JSON, headings, labels, explanations, negative prompts, markdown, or alternatives. Do not describe sound, a timeline, or camera motion. Output only the final single paragraph.`

const editRules = `You are an expert prompt writer for Krea 2 Identity Edit. The user supplies separate Change and Preserve instructions. Rewrite them as one concise English edit command. Make requested changes explicit while retaining every Preserve constraint, especially identity, facial features, hair, body proportions, clothing, pose, framing, background, lighting, text and untouched regions when named. Resolve ambiguity in favor of preservation and never invent an alteration that conflicts with Preserve. Do not repeat the source-image description as a new scene, and do not output headings, labels, JSON, negative prompts, explanations or alternatives. Aim for 45-100 words.`

func System(mode string, vision bool) string {
	if strings.EqualFold(mode, "t2i") {
		return imageRules
	}
	if strings.EqualFold(mode, "edit") {
		return editRules
	}
	prefix := t2vPrefix
	if strings.EqualFold(mode, "i2v") {
		prefix = i2vVisionPrefix
	}
	return prefix + "\n\n" + sharedRules
}
