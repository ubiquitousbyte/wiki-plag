package crawler

import "testing"

func TestLexer(t *testing.T) {
	tests := []struct {
		name   string
		tokens []token
		input  string
	}{
		{
			name:   "lex line feeds",
			input:  "\n\n\n\n\n",
			tokens: []token{{typ: tokenTypeEOF, data: ""}},
		},
		{
			name:  "lex latex",
			input: "asd{\\displaystyle T_{1}=T_{2}}asd",
			tokens: []token{
				{
					typ:  tokenTypeText,
					data: "asd",
				},
				{
					typ:  tokenTypeText,
					data: "asd",
				},
				{
					typ:  tokenTypeEOF,
					data: "",
				},
			},
		},
		{
			name: "lex 1",
			input: "Bei der Informatik handelt es sich um die Wissenschaft von der" +
				"systematischen Darstellung, Speicherung, Verarbeitung und Übertragung" +
				"von Informationen, wobei besonders die automatische Verarbeitung mit" +
				"Digitalrechnern betrachtet wird. Sie ist zugleich Grundlagen-" +
				"und Formalwissenschaft als auch Ingenieurdisziplin." +
				"\n\n\n== Geschichte der Informatik ==\n\n\n=== Ursprung ===\n",
			tokens: []token{
				{
					typ: tokenTypeText,
					data: "Bei der Informatik handelt es sich um die Wissenschaft von der" +
						"systematischen Darstellung, Speicherung, Verarbeitung und Übertragung" +
						"von Informationen, wobei besonders die automatische Verarbeitung mit" +
						"Digitalrechnern betrachtet wird. Sie ist zugleich Grundlagen-" +
						"und Formalwissenschaft als auch Ingenieurdisziplin.",
				},
				{
					typ:  tokenTypeTitle,
					data: "== Geschichte der Informatik ==",
				},
				{
					typ:  tokenTypeTitle,
					data: "=== Ursprung ===",
				},
				{
					typ:  tokenTypeEOF,
					data: "",
				},
			},
		},
		{
			name: "lex 2",
			input: "Unter einer Gleichung versteht man in der Mathematik eine " +
				"Aussage über die Gleichheit zweier Terme, die mit Hilfe des" +
				"Gleichheitszeichens („=“) symbolisiert wird." +
				"Formal hat eine Gleichung die Gestalt\n\n  \n    \n      \n" +
				"        \n          T\n          \n            1\n          \n" +
				"      \n        =\n" +
				"2\n{\\displaystyle T_{1}=T_{2}}\n" +
				" ,wobei der Term \n",
			tokens: []token{
				{
					typ: tokenTypeText,
					data: "Unter einer Gleichung versteht man in der Mathematik eine " +
						"Aussage über die Gleichheit zweier Terme, die mit Hilfe des" +
						"Gleichheitszeichens („=“) symbolisiert wird." +
						"Formal hat eine Gleichung die Gestalt",
				},
				{
					typ:  tokenTypeText,
					data: "  ",
				},

				{
					typ:  tokenTypeText,
					data: "    ",
				},

				{
					typ:  tokenTypeText,
					data: "      ",
				},

				{
					typ:  tokenTypeText,
					data: "        ",
				},

				{
					typ:  tokenTypeText,
					data: "          T",
				},

				{
					typ:  tokenTypeText,
					data: "          ",
				},

				{
					typ:  tokenTypeText,
					data: "            1",
				},

				{
					typ:  tokenTypeText,
					data: "          ",
				},

				{
					typ:  tokenTypeText,
					data: "      ",
				},

				{
					typ:  tokenTypeText,
					data: "        =",
				},

				{
					typ:  tokenTypeText,
					data: "2",
				},

				{
					typ:  tokenTypeText,
					data: " ,wobei der Term ",
				},

				{
					typ:  tokenTypeEOF,
					data: "",
				},
			},
		},
		{
			name: "lex 3",
			input: "ThinThread ist der Name eines Abhörprogramms, das von der" +
				"National Security Agency in den 1990er  von William Binney" +
				"entwickelt wurde.\nDas Programm wurde drei Wochen vor den" +
				"Terroranschlägen am 11. September 2001 eingestellt.\n" +
				"Es gilt jedoch als Blaupause für aktuelle Analysetechniken " +
				"der NSA.\n\n\n== Weblinks ==\nWilliam.Binney.HOPE.9.KEYNOTE.Part1," +
				"related to Thinthread development\nWilliam.Binney.HOPE.9.KEYNOTE.Part2," +
				"related to Thinthread development\nFilm on Thinthread and the SARC team\n" +
				"Dokumentation bei Phoenix: http://www.phoenix.de/content/phoenix/tv_programm/terrorjagd_im_netz/2563222",
			tokens: []token{
				{
					typ: tokenTypeText,
					data: "ThinThread ist der Name eines Abhörprogramms, das von der" +
						"National Security Agency in den 1990er  von William Binney" +
						"entwickelt wurde.",
				},
				{
					typ: tokenTypeText,
					data: "Das Programm wurde drei Wochen vor den" +
						"Terroranschlägen am 11. September 2001 eingestellt.",
				},
				{
					typ: tokenTypeText,
					data: "Es gilt jedoch als Blaupause für aktuelle Analysetechniken " +
						"der NSA.",
				},
				{
					typ:  tokenTypeTitle,
					data: "== Weblinks ==",
				},
				{
					typ: tokenTypeText,
					data: "William.Binney.HOPE.9.KEYNOTE.Part1," +
						"related to Thinthread development",
				},
				{
					typ: tokenTypeText,
					data: "William.Binney.HOPE.9.KEYNOTE.Part2," +
						"related to Thinthread development",
				},
				{
					typ:  tokenTypeText,
					data: "Film on Thinthread and the SARC team",
				},
				{
					typ:  tokenTypeText,
					data: "Dokumentation bei Phoenix: http://www.phoenix.de/content/phoenix/tv_programm/terrorjagd_im_netz/2563222",
				},
				{
					typ:  tokenTypeEOF,
					data: "",
				},
			},
		},
		{
			name:  "lex 4",
			input: "Cloud Computing\nDistributed Computing\nData",
			tokens: []token{
				{
					typ:  tokenTypeText,
					data: "Cloud Computing",
				},
				{
					typ:  tokenTypeText,
					data: "Distributed Computing",
				},
				{
					typ:  tokenTypeText,
					data: "Data",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			l := newLexer(test.input)
			for i := 0; i < len(test.tokens); i++ {
				tok := l.next()
				if tok != test.tokens[i] {
					t.Errorf("Expected token %v, but got %v instead", test.tokens[i], tok)
				}
			}
		})
	}
}
