module github.com/redis-developer/redis-vl-golang/extensions/vectorize/hf

go 1.25.0

require (
	github.com/redis-developer/redis-vl-golang v0.0.0
	github.com/sugarme/tokenizer v0.3.1-0.20251127070628-8dac234bfe56
	github.com/yalue/onnxruntime_go v1.31.0
)

require (
	github.com/emirpasic/gods v1.18.1 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/patrickmn/go-cache v2.1.0+incompatible // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	github.com/sugarme/regexpset v0.0.0-20200920021344-4d4ec8eaf93c // indirect
	golang.org/x/sync v0.14.0 // indirect
	golang.org/x/text v0.25.0 // indirect
)

// The hf provider lives in the same repository as the core module. Until
// both modules are published with release tags, resolve the core module
// locally. When tagging a release, tag both the root module (vX.Y.Z) and
// this one (extensions/vectorize/hf/vX.Y.Z), then update the require above
// and drop this replace.
replace github.com/redis-developer/redis-vl-golang => ../../..
