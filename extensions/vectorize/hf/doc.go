// Package hf provides a local (in-process) text vectorizer that runs
// Hugging Face sentence-transformer models through ONNX Runtime. It is the
// Go equivalent of Python RedisVL's HFTextVectorizer: give it a model name,
// it downloads the model files from the Hugging Face Hub on first use,
// caches them locally, and produces embeddings without any external API.
//
// The package is a separate Go module because it depends on ONNX Runtime
// via cgo. Importing it requires the onnxruntime shared library to be
// installed (https://onnxruntime.ai/docs/install/); the core
// redis-vl-golang module stays pure Go.
//
//	vec, err := hf.New(ctx, hf.Config{
//	    Model: "sentence-transformers/all-MiniLM-L6-v2",
//	})
//	if err != nil { ... }
//	defer vec.Close()
//	emb, err := vec.Embed(ctx, "Hello, world!")
//
// The returned vectorizer satisfies vectorize.Vectorizer, so it can be used
// anywhere the HTTP providers can: SemanticCache, SemanticMessageHistory,
// SemanticRouter, the MCP server, or cache.CachedVectorizer.
//
// Model requirements: the model repository must contain a tokenizer.json
// and an ONNX export (onnx/model.onnx by default; override with
// Config.ONNXFile for quantized variants). Pooling and normalization follow
// the model's sentence-transformers configuration (modules.json and
// 1_Pooling/config.json) so results match Python's sentence-transformers
// output.
package hf
