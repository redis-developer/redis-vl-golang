package hf

import (
	"fmt"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// loadTokenizer reads a Hugging Face tokenizer.json.
func loadTokenizer(path string) (*tokenizer.Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	tk, err := pretrained.FromReader(f)
	if err != nil {
		return nil, fmt.Errorf("hf: loading tokenizer.json: %w", err)
	}
	// The loader may install default padding (fixed-length) and truncation
	// strategies even when tokenizer.json specifies none. Both are handled
	// here instead: padding per batch to the longest row, truncation to the
	// model's max_seq_length.
	tk.WithPadding(nil)
	tk.WithTruncation(nil)
	return tk, nil
}

// tokenRow is the truncated (but not yet padded) token sequence of one text.
type tokenRow struct {
	ids     []int
	typeIDs []int
}

// rowFromEncoding converts one tokenizer encoding into a truncated token
// row: trailing tokenizer padding (attention mask 0) is dropped, and rows
// longer than maxSeq are cut while preserving a trailing special token such
// as [SEP], like Hugging Face truncation does.
func rowFromEncoding(enc *tokenizer.Encoding, maxSeq int) tokenRow {
	// Defensively drop any trailing padding the tokenizer produced
	// (attention mask 0); real tokens always carry mask 1.
	n := len(enc.Ids)
	if len(enc.AttentionMask) == n {
		for n > 0 && enc.AttentionMask[n-1] == 0 {
			n--
		}
	}
	row := tokenRow{
		ids:     append([]int(nil), enc.Ids[:n]...),
		typeIDs: append([]int(nil), enc.TypeIds[:min(n, len(enc.TypeIds))]...),
	}
	if len(row.typeIDs) != len(row.ids) {
		row.typeIDs = make([]int, len(row.ids))
	}

	if maxSeq > 0 && len(row.ids) > maxSeq {
		// Keep the trailing special token (e.g. [SEP] / EOS) if the
		// untruncated sequence ended with one.
		last := len(row.ids) - 1
		endsSpecial := len(enc.SpecialTokenMask) > last && enc.SpecialTokenMask[last] == 1
		lastID, lastType := row.ids[last], row.typeIDs[last]
		row.ids = row.ids[:maxSeq]
		row.typeIDs = row.typeIDs[:maxSeq]
		if endsSpecial {
			row.ids[maxSeq-1] = lastID
			row.typeIDs[maxSeq-1] = lastType
		}
	}
	return row
}

// tokenizeBatch encodes texts with special tokens and packs them into the
// padded matrices the ONNX model consumes.
func tokenizeBatch(tk *tokenizer.Tokenizer, texts []string, maxSeq, padID int) (*batchInput, error) {
	rows := make([]tokenRow, len(texts))
	for i, text := range texts {
		enc, err := tk.EncodeSingle(text, true)
		if err != nil {
			return nil, fmt.Errorf("hf: tokenizing text %d: %w", i, err)
		}
		rows[i] = rowFromEncoding(enc, maxSeq)
	}
	return buildBatch(rows, padID), nil
}

// tokenizePairs encodes (query, doc) pairs with special tokens (e.g.
// [CLS] query [SEP] doc [SEP] with segment ids 0/1) for cross-encoder
// scoring. Note: truncation cuts from the end of the pair (mostly the
// document), unlike Hugging Face's longest_first strategy; this only
// matters for pairs beyond the model's max sequence length.
func tokenizePairs(tk *tokenizer.Tokenizer, query string, docs []string, maxSeq, padID int) (*batchInput, error) {
	rows := make([]tokenRow, len(docs))
	for i, doc := range docs {
		enc, err := tk.EncodePair(query, doc, true)
		if err != nil {
			return nil, fmt.Errorf("hf: tokenizing pair %d: %w", i, err)
		}
		rows[i] = rowFromEncoding(enc, maxSeq)
	}
	return buildBatch(rows, padID), nil
}

// buildBatch pads all rows to the longest row with padID and flattens them
// row-major.
func buildBatch(rows []tokenRow, padID int) *batchInput {
	maxLen := 1 // never build zero-length tensors
	for _, row := range rows {
		if len(row.ids) > maxLen {
			maxLen = len(row.ids)
		}
	}

	in := &batchInput{
		batch:    len(rows),
		seqLen:   maxLen,
		ids:      make([]int64, len(rows)*maxLen),
		mask:     make([]int64, len(rows)*maxLen),
		typeIDs:  make([]int64, len(rows)*maxLen),
		rowMasks: make([][]int, len(rows)),
	}
	for b, row := range rows {
		offset := b * maxLen
		rowMask := make([]int, maxLen)
		for j, id := range row.ids {
			in.ids[offset+j] = int64(id)
			in.typeIDs[offset+j] = int64(row.typeIDs[j])
			in.mask[offset+j] = 1
			rowMask[j] = 1
		}
		for j := len(row.ids); j < maxLen; j++ {
			in.ids[offset+j] = int64(padID)
		}
		in.rowMasks[b] = rowMask
	}
	return in
}
