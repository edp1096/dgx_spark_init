APP := media
GO ?= go
NPM ?= npm

ifeq ($(OS),Windows_NT)
MKDIR_DIST := if not exist dist mkdir dist
CLEAN_DIST := if exist dist\$(APP).exe del /Q dist\$(APP).exe & if exist dist\$(APP)-* del /Q dist\$(APP)-*
CLEAN_WEB := if exist web\dist rmdir /S /Q web\dist
BUILD_LINUX_AMD64 := set CGO_ENABLED=0&& set GOOS=linux&& set GOARCH=amd64&& $(GO) build -trimpath -ldflags="-s -w" -o dist\$(APP)-linux-amd64 ./cmd/media
BUILD_LINUX_ARM64 := set CGO_ENABLED=0&& set GOOS=linux&& set GOARCH=arm64&& $(GO) build -trimpath -ldflags="-s -w" -o dist\$(APP)-linux-arm64 ./cmd/media
BUILD_WINDOWS_AMD64 := set CGO_ENABLED=0&& set GOOS=windows&& set GOARCH=amd64&& $(GO) build -trimpath -ldflags="-s -w" -o dist\$(APP)-windows-amd64.exe ./cmd/media
BUILD_WINDOWS_ARM64 := set CGO_ENABLED=0&& set GOOS=windows&& set GOARCH=arm64&& $(GO) build -trimpath -ldflags="-s -w" -o dist\$(APP)-windows-arm64.exe ./cmd/media
else
MKDIR_DIST := mkdir -p dist
CLEAN_DIST := rm -f dist/$(APP) dist/$(APP).exe dist/$(APP)-linux-* dist/$(APP)-windows-*.exe
CLEAN_WEB := rm -rf web/dist
BUILD_LINUX_AMD64 := CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build -trimpath -ldflags="-s -w" -o dist/$(APP)-linux-amd64 ./cmd/media
BUILD_LINUX_ARM64 := CGO_ENABLED=0 GOOS=linux GOARCH=arm64 $(GO) build -trimpath -ldflags="-s -w" -o dist/$(APP)-linux-arm64 ./cmd/media
BUILD_WINDOWS_AMD64 := CGO_ENABLED=0 GOOS=windows GOARCH=amd64 $(GO) build -trimpath -ldflags="-s -w" -o dist/$(APP)-windows-amd64.exe ./cmd/media
BUILD_WINDOWS_ARM64 := CGO_ENABLED=0 GOOS=windows GOARCH=arm64 $(GO) build -trimpath -ldflags="-s -w" -o dist/$(APP)-windows-arm64.exe ./cmd/media
endif

.PHONY: build web go-build test dist clean

build: web go-build

web:
	cd web && $(NPM) install && $(NPM) run build

go-build:
	$(GO) build ./...

test: web
	$(GO) test ./...

dist: web
	$(MKDIR_DIST)
	$(BUILD_LINUX_AMD64)
	$(BUILD_LINUX_ARM64)
	$(BUILD_WINDOWS_AMD64)
	$(BUILD_WINDOWS_ARM64)

clean:
	$(CLEAN_DIST)
	$(CLEAN_WEB)
