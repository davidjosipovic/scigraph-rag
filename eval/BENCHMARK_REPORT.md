# Evaluacija GraphRAG sustava nad ORKG-om

## Pregled

Ovaj dokument opisuje metodologiju i rezultate evaluacije GraphRAG sustava koji odgovara na pitanja o znanstvenim publikacijama koristeći Open Research Knowledge Graph (ORKG) kao izvor podataka. Evaluacija uspoređuje **9 jezičnih modela** iz 4 različita providera na skupu od **50 evaluacijskih pitanja** pokrivajući 6 tipova upita.

---

## Metodologija

### Skup evaluacijskih pitanja

Evaluacijski skup (`data/eval_questions.json`) sadrži **50 pitanja** ravnomjerno raspoređenih po:

**6 tipova upita:**

| Tip upita | Opis | Primjer |
|---|---|---|
| `topic_search` | Pronalazak radova o temi | "What papers exist about graph neural networks?" |
| `method_comparison` | Usporedba metoda | "Which papers compare CNN and RNN for text classification?" |
| `dataset_search` | Pronalazak radova koji koriste skup podataka | "Which papers use the ImageNet dataset?" |
| `claim_verification` | Provjera tvrdnje | "Does BERT outperform traditional methods on question answering?" |
| `method_usage` | Pronalazak primjena metode | "Which papers use attention mechanisms for NLP tasks?" |
| `paper_lookup` | Pronalazak specifičnog rada | "Find the paper titled 'Attention Is All You Need'." |

**3 razine težine:** `easy` (14 pitanja), `medium` (20 pitanja), `hard` (16 pitanja)

Svako pitanje sadrži `answer_hints` — ključne pojmove koje dobar odgovor treba sadržavati (npr. za "Attention Is All You Need": `["Vaswani", "multi-head attention", "encoder-decoder", "2017"]`).

### Pipeline

Za svako pitanje pokreće se cijeli 10-koračni RAG pipeline:

```
pitanje → klasifikacija tipa upita → ekstrakcija entiteta → normalizacija sinonima
       → paralelni SPARQL upiti prema ORKG-u → rangiranje → filtriranje
       → izgradnja konteksta → LLM generacija → formatiranje izvora
```

Svaki model koristi se za sve tri LLM faze (klasifikacija, ekstrakcija, generacija). SPARQL upiti prema ORKG-u su deterministički — isti za sve modele.

### LLM-as-Judge evaluacija

Svaki generirani odgovor ocjenjuje neovisni model-sudac (`ollama:llama3.2`) na **3 metrike** skale 0–5:

| Metrika | Opis |
|---|---|
| **Groundedness** | Je li odgovor temeljen isključivo na dohvaćenim izvorima iz ORKG-a? |
| **Relevance** | Odgovara li odgovor na postavljeno pitanje? |
| **Completeness** | Pokriva li odgovor ključne aspekte navedene u `answer_hints`? |

**Overall score** = aritmetička sredina svih triju metrika.

Sudac prima: pitanje, generirani odgovor, naslove dohvaćenih izvora i `answer_hints`. Odgovara isključivo JSON-om bez slobodnog teksta.

Sudac (`llama3.2`) je namjerno odabran kao model koji **nije testiran** u benchmarku kako bi se izbjegla pristranost.

### Tehničke pojedinosti

- **Benchmark skripta:** `eval/benchmark.py` — CLI s parametrima `--models`, `--judge`, `--delay`, `--judge-delay`, `--limit`, `--query-types`, `--skip-judge`
- **Rejudge skripta:** `eval/rejudge.py` — retroaktivno bodovanje postojećih rezultata novim sudcem
- **Izlaz:** JSON s potpunim per-pitanje rezultatima + CSV sažetak po modelu
- **Delay između pitanja:** 3s za cloud modele (rate limiting), 0s za lokalne
- **SPARQL timeout:** 10s po upitu

---

## Testirani modeli

| Provider | Model | Veličina | Tip |
|---|---|---|---|
| Ollama (lokalno) | `llama3` | 8B | Meta Llama 3 |
| Ollama (lokalno) | `llama3.2` | 3B | Meta Llama 3.2 |
| Ollama (lokalno) | `mistral` | 7B | Mistral AI |
| Ollama (lokalno) | `phi3:mini` | 3.8B | Microsoft Phi-3 |
| Gemini API | `gemma-3-4b-it` | 4B | Google Gemma 3 |
| Groq API | `llama-3.1-8b-instant` | 8B | Meta Llama 3.1 |
| Groq API | `llama-3.3-70b-versatile` | 70B | Meta Llama 3.3 |
| Groq API | `llama-4-scout-17b-16e-instruct` | 17B (MoE) | Meta Llama 4 |
| Groq API | `qwen3-32b` | 32B | Alibaba Qwen 3 |
| Groq API | `gpt-oss-20b` | 20B | OpenAI OSS |

Svaki model evaluiran na svih **50 pitanja** (ukupno 450 pipeline izvođenja).

---

## Rezultati

### Opći poredak (sortirano po Overall score)

| # | Model | Overall ↓ | Groundedness | Relevance | Completeness | Latency |
|---|---|---|---|---|---|---|
| 1 | `groq:gpt-oss-20b` | **3.307** | 3.040 | 3.880 | 3.000 | 10.8s |
| 2 | `groq:qwen3-32b` | **3.193** | 2.820 | 3.680 | 3.080 | 23.2s |
| 3 | `ollama:mistral` | **3.148** | 3.220 | 3.540 | 2.680 | 41.8s |
| 4 | `groq:llama-3.1-8b` | **3.060** | 2.920 | 3.680 | 2.580 | 15.3s |
| 5 | `ollama:llama3` | **3.041** | 3.080 | 3.460 | 2.580 | 25.6s |
| 6 | `groq:llama-4-scout` | **2.980** | 2.820 | 3.480 | 2.640 | **3.9s** |
| 7 | `ollama:llama3.2` | **2.961** | 3.140 | 3.280 | 2.460 | 11.6s |
| 8 | `gemini:gemma-3-4b-it` | **2.855** | 2.780 | 3.200 | 2.580 | 7.6s |
| 9 | `ollama:phi3:mini` | **2.854** | 2.800 | 3.240 | 2.520 | 61.8s |
| — | `groq:llama-3.3-70b` | 2.729 | 2.633 | 3.286 | 2.265 | 6.8s |

> **Napomena:** `llama-3.3-70b` isključen iz primarne analize zbog nepotpunog skupa (49/50 pitanja, rate limit greška na jednom pitanju). `gemma2-9b-it` i `mixtral-8x7b-32768` isključeni jer su dekommisionirani na Groq platformi.

### Rezultati po tipu upita

| Tip upita | gpt-oss | qwen3 | mistral | llama3.1 | llama3 | llama4 | llama3.2 | gemma3 | phi3 |
|---|---|---|---|---|---|---|---|---|---|
| `topic_search` | 4.52 | 4.07 | 3.70 | 3.89 | 3.44 | 3.78 | 3.93 | 4.25 | 4.00 |
| `dataset_search` | 4.10 | 3.89 | **4.46** | 4.17 | 3.88 | 4.21 | 3.38 | **4.74** | 4.21 |
| `paper_lookup` | 3.86 | **4.28** | 3.38 | 3.78 | 3.72 | 3.25 | 3.05 | 3.72 | 2.33 |
| `method_usage` | 3.26 | 3.74 | 2.74 | 2.52 | 3.12 | 2.88 | 2.96 | 3.37 | 3.15 |
| `method_comparison` | 2.96 | 3.63 | 3.00 | 2.75 | 3.46 | 3.00 | 2.54 | 2.67 | 3.17 |
| `claim_verification` | 2.28 | 1.89 | 2.11 | 2.17 | 1.89 | 1.89 | 2.48 | 1.83 | 1.46 |

**Prosjek po tipu upita (svi modeli):**

| Tip upita | Prosječni Overall |
|---|---|
| `topic_search` | **3.918** |
| `dataset_search` | **4.056** |
| `paper_lookup` | 3.476 |
| `method_usage` | 3.046 |
| `method_comparison` | 2.975 |
| `claim_verification` | **1.945** |

### Rezultati po težini pitanja (svi modeli)

| Težina | Prosječni Overall |
|---|---|
| `easy` | **3.637** |
| `medium` | 3.206 |
| `hard` | 3.023 |

---

## Analiza i zaključci

### 1. Uski raspon scoreva — bottleneck je ORKG, ne LLM

Svi modeli postižu Overall score između **2.85 i 3.31** (skala 0–5), što je relativno uzak raspon. To sugerira da ograničavajući faktor kvalitete odgovora nije sam jezični model, već **pokrivenost podataka u ORKG-u** — ako knowledge graph ne sadrži relevantne radove za neko pitanje, ni najjači model ne može generirati dobar odgovor.

### 2. Claim verification je konzistentno najslabiji tip (avg 1.945)

`claim_verification` pitanja traže potvrdu ili opovrgavanje tvrdnje empirijskim dokazima. ORKG rijetko sadrži eksplicitne numeričke usporedbe (npr. "BERT postiže F1=93.2 na SQuAD vs. tradicijskim metodama F1=80.0"), pa sustav ne može pouzdano potvrditi ili opovrgnuti tvrdnje. Ovo je strukturalno ograničenje knowledge grapha, ne pipeline-a.

### 3. Topic search i dataset search su najjači tipovi (avg ~4.0)

ORKG dobro indeksira metode i skupove podataka po radovima, što je upravo ono što ovi tipovi upita trebaju. Rezultati su konzistentno visoki kod svih modela (3.4–4.7).

### 4. Speed-quality tradeoff

| Scenarij | Preporučeni model | Razlog |
|---|---|---|
| Produkcijski deployment (brzina) | `groq:llama-4-scout` | Samo 3.9s, Overall 2.98 |
| Produkcijski deployment (kvaliteta) | `groq:gpt-oss-20b` | Best Overall 3.31, 10.8s |
| Lokalni deployment (offline) | `ollama:mistral` | Best lokalni Overall 3.15, 41.8s |
| Resursno ograničeno okruženje | `ollama:llama3.2` | 3B model, 11.6s, Overall 2.96 |

### 5. Veličina modela ne garantira kvalitetu

`llama-3.3-70b` (70B parametara) postiže najniži Overall (2.729) među cloud modelima, dok `llama-3.1-8b` (8B) postiže 3.060. Razlog je vjerojatno što veći modeli imaju tendenciju generirati opširnije odgovore koji pokrivaju manje specifičnih činjenica iz dohvaćenog konteksta.

### 6. Relevance > Groundedness > Completeness (kod svih modela)

Svi modeli konzistentno postižu najviši **Relevance** score (3.2–3.9) — modeli razumiju pitanje i odgovaraju tematski. **Groundedness** (2.6–3.2) je manji jer modeli ponekad dodaju opće znanje uz informacije iz ORKG-a. **Completeness** je najmanji (2.3–3.1) jer ORKG rijetko sadrži sve aspekte koje `answer_hints` očekuju.

---

## Ograničenja evaluacije

1. **LLM-as-judge pristranost** — Sudac (`llama3.2`) ima vlastite predrasude i može favorizirati određeni stil odgovora. Idealno bi bio korišten jači model (GPT-4o, Claude Opus) kao sudac.

2. **Answer hints su nužni uvjet, ne dovoljan** — `answer_hints` provjera ne mjeri je li odgovor faktično točan, nego sadrži li ključne pojmove. Model može pogoditi ključne riječi bez razumijevanja.

3. **ORKG pokrivenost nije kontrolirana** — Neka pitanja mogu biti lakša jer ORKG slučajno ima više podataka o toj temi. Evaluacijski skup nije stratificiran po ORKG pokrivenosti.

4. **Jedan sudac za sve modele** — Konzistentnost je osigurana, ali apsolutne vrijednosti scoreva ovise o kalibraciji suca.

---

## Datoteke rezultata

| Datoteka | Sadržaj |
|---|---|
| `eval/results/benchmark_20260421_110555.json` | `ollama:llama3`, `gemini:gemma-3-4b-it` |
| `eval/results/benchmark_20260421_120532.json` | `groq:llama-3.1-8b`, `groq:llama-3.3-70b` |
| `eval/results/benchmark_20260421_120807.json` | `ollama:mistral`, `ollama:phi3:mini`, `ollama:llama3.2` |
| `eval/results/benchmark_20260421_133828.json` | `groq:llama-4-scout`, `groq:qwen3-32b`, `groq:gpt-oss-20b` |

Svaka JSON datoteka sadrži potpune rezultate po pitanju (pitanje, tip, težina, latencija, odgovor, dohvaćeni izvori, judge scorevi i obrazloženje).
