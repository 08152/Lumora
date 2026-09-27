/* LUMORA Tokenizer - no external dependencies */
(function (global) {
  'use strict';

  class LUMORATokenizer {
    constructor() {
      this.specialTokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
      this.tokenToId = new Map();
      this.idToToken = [];
      this.reset();
    }

    reset() {
      this.tokenToId.clear();
      this.idToToken.length = 0;
      for (const token of this.specialTokens) this.addToken(token);
    }

    addToken(token) {
      if (!this.tokenToId.has(token)) {
        const id = this.idToToken.length;
        this.tokenToId.set(token, id);
        this.idToToken.push(token);
      }
      return this.tokenToId.get(token);
    }

    /**
     * Tokenizes words, numbers, punctuation and common emoji/Unicode symbols.
     * Example: "Hallo LUMORA!" -> ["Hallo", "LUMORA", "!"]
     */
    split(text) {
      return String(text ?? '').match(/\p{L}[\p{L}\p{M}\p{N}_'-]*|\p{N}+(?:[.,]\p{N}+)*|[^\s\p{L}\p{N}]/gu) || [];
    }

    buildFromData(data) {
      this.reset();
      const visit = (value) => {
        if (typeof value === 'string') {
          for (const token of this.split(value)) this.addToken(token);
        } else if (Array.isArray(value)) {
          value.forEach(visit);
        } else if (value && typeof value === 'object') {
          Object.keys(value).forEach(key => {
            for (const token of this.split(key)) this.addToken(token);
            visit(value[key]);
          });
        }
      };
      visit(data);
      this.addToken('<UNK>');
      return this;
    }

    encode(text) {
      return this.split(text).map(token => this.tokenToId.has(token) ? this.tokenToId.get(token) : this.tokenToId.get('<UNK>'));
    }

    decode(ids) {
      return (Array.isArray(ids) ? ids : []).map(id => this.idToToken[id] ?? '<UNK>');
    }

    decodeText(ids) {
      const tokens = this.decode(ids).filter(t => !this.specialTokens.includes(t));
      return LUMORATokenizer.detokenize(tokens);
    }

    static detokenize(tokens) {
      let out = '';
      const noSpaceBefore = new Set(['.', ',', '!', '?', ':', ';', ')', ']', '}', '%', '…']);
      const noSpaceAfter = new Set(['(', '[', '{', '„', '«', '“']);
      for (const token of tokens) {
        if (!out) out = token;
        else if (noSpaceBefore.has(token) || token.startsWith("'")) out += token;
        else if (noSpaceAfter.has(out.slice(-1))) out += token;
        else out += ' ' + token;
      }
      return out.replace(/\s+([,.!?;:])/g, '$1').trim();
    }

    get vocabSize() {
      return this.idToToken.length;
    }
  }

  global.LUMORATokenizer = LUMORATokenizer;
})(globalThis);
