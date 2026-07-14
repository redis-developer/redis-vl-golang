package query

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/redis/redis-vl-golang/filter"
)

// TextQuery runs a full-text search with an optional filter expression
// (port of redisvl.query.TextQuery).
type TextQuery struct {
	opts Options

	text         string
	fieldWeights map[string]float64
	textWeights  map[string]float64
	stopwords    map[string]bool
	filter       *filter.Expression
	params       map[string]any
}

// NewTextQuery creates a full-text query for text against a single field,
// with English stopwords and the BM25STD scorer by default.
func NewTextQuery(text string, textFieldName string) *TextQuery {
	return &TextQuery{
		text:         text,
		fieldWeights: map[string]float64{textFieldName: 1.0},
		stopwords:    englishStopwords(),
		opts: Options{
			Num:        10,
			Dialect:    2,
			Scorer:     "BM25STD",
			WithScores: true,
		},
	}
}

// FieldWeights searches multiple text fields with per-field weights.
func (q *TextQuery) FieldWeights(weights map[string]float64) *TextQuery {
	q.fieldWeights = weights
	return q
}

// TextWeights boosts individual words within the query text.
func (q *TextQuery) TextWeights(weights map[string]float64) *TextQuery {
	q.textWeights = map[string]float64{}
	for word, w := range weights {
		q.textWeights[strings.ToLower(strings.TrimSpace(word))] = w
	}
	return q
}

// Scorer sets the text scoring algorithm (TFIDF, BM25STD, BM25,
// TFIDF.DOCNORM, DISMAX, DOCSCORE). Default BM25STD.
func (q *TextQuery) Scorer(scorer string) *TextQuery { q.opts.Scorer = scorer; return q }

// Stopwords replaces the stopword set removed from the query text
// client-side. Pass nothing to disable stopword removal.
func (q *TextQuery) Stopwords(words ...string) *TextQuery {
	q.stopwords = map[string]bool{}
	for _, w := range words {
		q.stopwords[w] = true
	}
	return q
}

// Filter applies a filter expression alongside the text search.
func (q *TextQuery) Filter(f *filter.Expression) *TextQuery { q.filter = f; return q }

// ReturnFields sets the fields to return with results.
func (q *TextQuery) ReturnFields(fields ...string) *TextQuery {
	q.opts.ReturnFields = fields
	return q
}

// NumResults sets the top-k results to return (default 10).
func (q *TextQuery) NumResults(n int) *TextQuery { q.opts.Num = n; return q }

// SortByField sorts results by the given field (default: text score).
func (q *TextQuery) SortByField(field string, asc bool) *TextQuery {
	q.opts.SortBy = &SortBy{Field: field, Asc: asc}
	return q
}

// Dialect sets the query dialect (default 2).
func (q *TextQuery) Dialect(d int) *TextQuery { q.opts.Dialect = d; return q }

// InOrder requires query terms to appear in order.
func (q *TextQuery) InOrder() *TextQuery { q.opts.InOrder = true; return q }

// NoReturnScore disables WITHSCORES.
func (q *TextQuery) NoReturnScore() *TextQuery { q.opts.WithScores = false; return q }

// WithParams attaches PARAMS to the query.
func (q *TextQuery) WithParams(params map[string]any) *TextQuery {
	q.params = params
	return q
}

// tokenizeAndEscape converts raw user text into a Redis full-text query
// joined by ORs, removing stopwords and applying per-word weights.
func (q *TextQuery) tokenizeAndEscape(userQuery string) string {
	return tokenizeAndEscape(userQuery, q.stopwords, q.textWeights)
}

// tokenizeAndEscape is shared by TextQuery and the hybrid/aggregation
// queries (port of redisvl.utils.full_text_query_helper).
func tokenizeAndEscape(userQuery string, stopwords map[string]bool, textWeights map[string]float64) string {
	var tokens []string
	for _, token := range strings.Fields(userQuery) {
		t := strings.TrimSpace(token)
		t = strings.Trim(t, ",")
		t = strings.ReplaceAll(t, "“", "") // left curly quote
		t = strings.ReplaceAll(t, "”", "") // right curly quote
		t = strings.ToLower(t)
		t = filter.EscapeToken(t, false)
		if t == "" || stopwords[t] {
			continue
		}
		if w, ok := textWeights[t]; ok {
			t = fmt.Sprintf("%s=>{$weight:%s}", t, strconv.FormatFloat(w, 'g', -1, 64))
		}
		tokens = append(tokens, t)
	}
	return strings.Join(tokens, " | ")
}

// fullTextQueryString builds "(~@field:(tokens) [AND filter])" — the shared
// text+filter clause used by hybrid queries (port of
// FullTextQueryHelper.build_query_string; note the optional-match ~).
func fullTextQueryString(text, textFieldName string, fe *filter.Expression, stopwords map[string]bool, textWeights map[string]float64) string {
	q := "(~@" + textFieldName + ":(" + tokenizeAndEscape(text, stopwords, textWeights) + ")"
	fs := filterString(fe)
	if fs != "" && fs != "*" {
		q += " AND " + fs
	}
	return q + ")"
}

// QueryString implements Query.
func (q *TextQuery) QueryString() string {
	escaped := q.tokenizeAndEscape(q.text)

	// deterministic field ordering
	fields := make([]string, 0, len(q.fieldWeights))
	for f := range q.fieldWeights {
		fields = append(fields, f)
	}
	sort.Strings(fields)

	var fieldQueries []string
	for _, f := range fields {
		w := q.fieldWeights[f]
		if w == 1.0 {
			fieldQueries = append(fieldQueries, fmt.Sprintf("@%s:(%s)", f, escaped))
		} else {
			fieldQueries = append(fieldQueries,
				fmt.Sprintf("@%s:(%s) => { $weight: %s }", f, escaped,
					strconv.FormatFloat(w, 'g', -1, 64)))
		}
	}

	var text string
	if len(fieldQueries) == 1 {
		text = fieldQueries[0]
	} else {
		text = "(" + strings.Join(fieldQueries, " | ") + ")"
	}

	fe := filterString(q.filter)
	if fe != "" && fe != "*" {
		text += " AND " + fe
	}
	return text
}

// Err returns any deferred filter construction error.
func (q *TextQuery) Err() error { return q.filter.Err() }

// Params implements Query.
func (q *TextQuery) Params() map[string]any { return q.params }

// Options implements Query.
func (q *TextQuery) Options() *Options { return &q.opts }

// englishStopwords returns the default English stopword set (equivalent to
// nltk's english corpus, which the Python library downloads at runtime).
func englishStopwords() map[string]bool {
	words := []string{
		"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
		"your", "yours", "yourself", "yourselves", "he", "him", "his",
		"himself", "she", "her", "hers", "herself", "it", "its", "itself",
		"they", "them", "their", "theirs", "themselves", "what", "which",
		"who", "whom", "this", "that", "these", "those", "am", "is", "are",
		"was", "were", "be", "been", "being", "have", "has", "had", "having",
		"do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
		"or", "because", "as", "until", "while", "of", "at", "by", "for",
		"with", "about", "against", "between", "into", "through", "during",
		"before", "after", "above", "below", "to", "from", "up", "down",
		"in", "out", "on", "off", "over", "under", "again", "further",
		"then", "once", "here", "there", "when", "where", "why", "how",
		"all", "any", "both", "each", "few", "more", "most", "other",
		"some", "such", "no", "nor", "not", "only", "own", "same", "so",
		"than", "too", "very", "s", "t", "can", "will", "just", "don",
		"should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
		"aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
		"ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
		"weren", "won", "wouldn",
	}
	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[w] = true
	}
	return set
}
