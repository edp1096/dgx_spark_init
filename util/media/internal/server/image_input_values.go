package server

func firstImagePath(paths []string) string {
	if len(paths) > 0 {
		return paths[0]
	}
	return ""
}
