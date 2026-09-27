/*
 * LUMORA Engine
 *
 * This is deliberately honest: it is not a large neural transformer.
 * It is a small, fully local autoregressive word n-gram language model with
 * context retrieval, rule conditioning and sampling. It actually generates
 * text rather than selecting a single canned answer.
 */
(function (global) {
  'use strict';

  class LocalNGramLanguageModel {
    constructor(options = {}) {
      this.order = Math.max(2, Math.min(4, options.order || 3));
      this.counts = new Map();
      this.unigrams = new Map();
      this.starts = new Map();
      this.sentences = [];
      this.totalTokens = 0;
    }

    _key(tokens) { return tokens.join('\u0001'); }

    train(sentences) {
      this.counts.clear();
      this.unigrams.clear();
      this.starts.clear();
      this.sentences = [];
      this.totalTokens = 0;

      for (const raw of sentences || []) {
        const sentence = String(raw ?? '').replace(/\s+/g, ' ').trim();
        if (!sentence) continue;
        const tokens = this.tokenizeWords(sentence);
        if (!tokens.length) continue;
        this.sentences.push(sentence);
        const seq = ['<BOS>', '<BOS>', ...tokens, '<EOS>'];
        for (const t of tokens) {
          this.unigrams.set(t, (this.unigrams.get(t) || 0) + 1);
          this.totalTokens++;
        }
        const first = tokens.slice(0, Math.min(3, tokens.length)).join(' ');
        this.starts.set(first, (this.starts.get(first) || 0) + 1);
        for (let i = 2; i < seq.length; i++) {
          for (let n = 2; n <= this.order; n++) {
            const start = Math.max(0, i - n + 1);
            const context = seq.slice(start, i);
            const next = seq[i];
            const key = this._key(context);
            if (!this.counts.has(key)) this.counts.set(key, new Map());
            const bucket = this.counts.get(key);
            bucket.set(next, (bucket.get(next) || 0) + 1);
          }
        }
      }
      return this;
    }

    tokenizeWords(text) {
      return String(text ?? '').match(/\p{L}[\p{L}\p{M}\p{N}_'-]*|\p{N}+(?:[.,]\p{N}+)*|[^\s\p{L}\p{N}]/gu) || [];
    }

    _sampleDistribution(bucket, temperature = 0.95) {
      const entries = [...bucket.entries()].filter(([t]) => t !== '<BOS>');
      if (!entries.length) return null;
      const powered = entries.map(([token, count]) => [token, Math.pow(count, 1 / Math.max(0.15, temperature))]);
      const sum = powered.reduce((a, [, w]) => a + w, 0);
      let r = Math.random() * sum;
      for (const [token, weight] of powered) {
        r -= weight;
        if (r <= 0) return token;
      }
      return powered[powered.length - 1][0];
    }

    generate(seedText = '', maxTokens = 42, temperature = 0.92) {
      if (!this.sentences.length) return '';
      let seed = this.tokenizeWords(seedText).slice(-2);
      if (!seed.length) {
        const pool = [...this.starts.entries()];
        seed = this.tokenizeWords(pool[Math.floor(Math.random() * pool.length)][0]).slice(0, 2);
      }

      const output = [];
      let context = ['<BOS>', ...seed];
      for (let i = 0; i < maxTokens; i++) {
        let next = null;
        for (let n = Math.min(this.order - 1, context.length); n >= 1 && !next; n--) {
          const key = this._key(context.slice(-n));
          const bucket = this.counts.get(key);
          if (bucket) next = this._sampleDistribution(bucket, temperature);
        }
        if (!next || next === '<EOS>') break;
        output.push(next);
        context.push(next);
      }
      return this.detokenize(output);
    }

    detokenize(tokens) {
      let out = '';
      const noSpaceBefore = new Set(['.', ',', '!', '?', ':', ';', ')', ']', '}', '%', '…']);
      const noSpaceAfter = new Set(['(', '[', '{', '„', '«', '“']);
      for (const token of tokens) {
        if (!out) out = token;
        else if (noSpaceBefore.has(token) || token.startsWith("'")) out += token;
        else if (noSpaceAfter.has(out.slice(-1))) out += token;
        else out += ' ' + token;
      }
      return out.trim();
    }
  }

  function collectStrings(value, out = []) {
    if (typeof value === 'string') out.push(value);
    else if (Array.isArray(value)) value.forEach(v => collectStrings(v, out));
    else if (value && typeof value === 'object') Object.values(value).forEach(v => collectStrings(v, out));
    return out;
  }

  function normalize(s) {
    return String(s ?? '').toLocaleLowerCase('de-DE').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  }

  function words(s) {
    return normalize(s).match(/[a-z0-9äöüß]+/g) || [];
  }

  class LUMORAEngine {
    constructor(data = {}, tokenizer = null) {
      this.data = data;
      this.tokenizer = tokenizer;
      this.languageModel = new LocalNGramLanguageModel({ order: 3 });
      this.history = [];
      this.maxHistory = 12;
      this.lastIntent = null;
      this.train();
    }

    setData(data) {
      this.data = data || {};
      this.train();
    }

    train() {
      const corpus = collectStrings(this.data).filter(s => s.trim().length >= 2);
      const expanded = [...corpus];
      const patterns = this.data?.sprechen?.muster || this.data?.muster || {};
      for (const [key, values] of Object.entries(patterns)) {
        if (Array.isArray(values)) {
          for (const v of values) expanded.push(`${key}: ${v}`);
        }
      }
      this.languageModel.train(expanded);
      return {
        sentenceCount: this.languageModel.sentences.length,
        tokenCount: this.languageModel.totalTokens,
        vocabularySize: this.tokenizer?.vocabSize ?? 0
      };
    }

    learn(text) {
      const learned = String(text ?? '').trim();
      if (!learned) return false;
      if (!this.data.sprechen) this.data.sprechen = {};
      if (!Array.isArray(this.data.sprechen.lernsaetze)) this.data.sprechen.lernsaetze = [];
      this.data.sprechen.lernsaetze.push(learned);
      this.train();
      return true;
    }

    _patternEntries() {
      const patterns = this.data?.sprechen?.muster || this.data?.muster || {};
      return Object.entries(patterns).map(([key, responses]) => ({ key, responses: Array.isArray(responses) ? responses : [] }));
    }

    _findIntent(input) {
      const inputWords = new Set(words(input));
      let best = null;
      for (const entry of this._patternEntries()) {
        const keyWords = words(entry.key);
        let score = 0;
        for (const w of keyWords) if (inputWords.has(w)) score += 2;
        for (const w of inputWords) if (keyWords.includes(w)) score += 0.35;
        if (score > (best?.score || 0)) best = { ...entry, score };
      }
      const rules = this.data?.regeln?.regeln || [];
      for (const rule of rules) {
        const triggers = Array.isArray(rule.wenn) ? rule.wenn : [];
        if (triggers.some(t => inputWords.has(normalize(t)) || normalize(input).includes(normalize(t)))) {
          if (!best || best.score < 1) best = { key: rule.name || 'regel', responses: [rule.aktion || 'Ich beachte die Regel.'], score: 1.05, rule };
          else best.rule = rule;
        }
      }
      return best;
    }

    _knowledgeMatches(input) {
      const inputWords = new Set(words(input));
      const knowledge = this.data?.sprechen?.wissen || [];
      return knowledge.map(item => {
        const text = typeof item === 'string' ? item : `${item.thema || ''} ${item.inhalt || item.text || ''}`;
        const ws = words(text);
        let score = 0;
        for (const w of ws) if (inputWords.has(w)) score++;
        return { item, text, score };
      }).filter(x => x.score > 0).sort((a, b) => b.score - a.score).slice(0, 3);
    }

    _personalityPrefix() {
      const p = this.data?.persoenlichkeit || this.data?.['persöhnlichkeit'] || this.data?.persönlichkeit || {};
      const traits = Array.isArray(p.eigenschaften) ? p.eigenschaften : [];
      if (!traits.length) return '';
      const style = p.stil?.sprache ? ` Sprache: ${p.stil.sprache}.` : '';
      return `${p.beschreibung || ''}${style}`.trim();
    }

    _contextSeed() {
      const recent = this.history.slice(-4).map(m => m.text).join(' ');
      return recent;
    }

    _selectFragment(intent) {
      if (!intent?.responses?.length) return '';
      return intent.responses[Math.floor(Math.random() * intent.responses.length)];
    }

    _compose(input, intent, matches) {
      const generatedSeed = [this._contextSeed(), input, intent?.key || '', matches[0]?.text || ''].join(' ');
      let generated = this.languageModel.generate(generatedSeed, 28, 0.85 + Math.random() * 0.25);
      generated = generated.replace(/<BOS>|<EOS>/g, '').trim();

      const known = this._selectFragment(intent);
      const fact = matches[0]?.item;
      let factText = '';
      if (fact) factText = typeof fact === 'string' ? fact : (fact.inhalt || fact.text || '');

      // Compose new text from data fragments + generated continuation.
      if (known && generated) {
        const g = generated.charAt(0).toLowerCase() + generated.slice(1);
        return `${known.replace(/[.!?]+$/, '')}. ${g}`.trim();
      }
      if (factText && generated) {
        const g = generated.charAt(0).toLowerCase() + generated.slice(1);
        return `${factText.replace(/[.!?]+$/, '')}. ${g}`.trim();
      }
      if (known) return known;
      if (factText) return factText;
      if (generated) return generated;

      return 'Ich habe dazu noch wenig Daten. Füge weitere Inhalte in DATEN/ ein, damit ich daraus mehr lernen kann.';
    }

    generate(input) {
      const userText = String(input ?? '').trim();
      if (!userText) return 'Schreib mir etwas, dann kann ich daraus eine Antwort erzeugen.';
      const intent = this._findIntent(userText);
      const matches = this._knowledgeMatches(userText);
      const response = this._compose(userText, intent, matches);
      this.lastIntent = intent;
      this.history.push({ role: 'user', text: userText }, { role: 'assistant', text: response });
      if (this.history.length > this.maxHistory) this.history.splice(0, this.history.length - this.maxHistory);
      return response;
    }

    encode(text) {
      return this.tokenizer ? this.tokenizer.encode(text) : [];
    }

    decode(ids) {
      return this.tokenizer ? this.tokenizer.decodeText(ids) : '';
    }

    stats() {
      return {
        vocabSize: this.tokenizer?.vocabSize ?? 0,
        trainingSentences: this.languageModel.sentences.length,
        trainingTokens: this.languageModel.totalTokens,
        historyMessages: this.history.length
      };
    }
  }

  global.LUMORAEngine = LUMORAEngine;
})(globalThis);
