package prompt

import "strings"

const sharedRules = `Write a single, highly detailed audio-visual caption in the style of video-model training captions. Preserve every element the user stated and expand faithfully without contradiction.

Begin immediately with action or visible detail. Use objective, observable descriptions. Include the environment, materials, textures, lighting, colors, subject appearance and spatial positions. Include exactly one appropriate shot type, the camera motion (or explicitly state that it remains static), and the camera viewpoint, woven naturally into the prose. Describe a complete soundscape including exact original-language dialogue when supplied. Keep events chronological and in real time. Output one continuous paragraph of roughly 150-220 words with no heading, label, JSON, or preamble. If the request is not English, write the caption in English. Use cinematic lighting, film-grade color and contrast, crisp detail, pleasing composition and depth without adding actions or objects that contradict the request.`

const t2vPrefix = `You are given a user's short text-to-video request. Produce the detailed caption that best fulfills it. The generated video is scored against the ORIGINAL request, so do not omit or replace any requested element.`

const i2vVisionPrefix = `You are given a REFERENCE IMAGE, which is the exact first frame, and a user's short image-to-video request. The caption must begin from that image with the same subjects, appearance, setting, lighting, viewpoint and composition, then describe the requested motion as one continuous take. Never contradict or invent details inconsistent with the image.`

func System(mode string, vision bool) string {
	prefix := t2vPrefix
	if strings.EqualFold(mode, "i2v") {
		prefix = i2vVisionPrefix
	}
	return prefix + "\n\n" + sharedRules
}
