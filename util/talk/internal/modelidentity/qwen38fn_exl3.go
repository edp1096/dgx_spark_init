package modelidentity

const Qwen38FNEXL3 = "qwen38fn_exl3"

// Previous release names are accepted only to migrate persisted references.
const LegacyQwenRuntime = "flash-next-native-exl3"
const LegacyQwenModel = "huihui-qwen38-native-exl3"
const LegacyQwenType = "qwen3.8-exl3"
const LegacyQwenCompose = "compose.flash-next-native-exl3.yaml"
const LegacyQwenContainer = "sparktalk-qwen38-native-exl3"

func CanonicalID(value string) string {
	switch value {
	case LegacyQwenRuntime, LegacyQwenModel, LegacyQwenType:
		return Qwen38FNEXL3
	}
	return value
}

func CanonicalContainer(value string) string {
	if value == LegacyQwenContainer {
		return "sparktalk-" + Qwen38FNEXL3
	}
	return value
}

func CanonicalCompose(value string) string {
	if value == LegacyQwenCompose {
		return "compose." + Qwen38FNEXL3 + ".yaml"
	}
	return value
}
