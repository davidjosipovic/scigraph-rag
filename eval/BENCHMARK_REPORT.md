# Evaluacija GraphRAG sustava nad ORKG-om

## Pregled

Ovaj dokument opisuje metodologiju i rezultate evaluacije GraphRAG sustava koji odgovara na pitanja o znanstvenim publikacijama koristeći Open Research Knowledge Graph (ORKG) kao izvor podataka. Evaluacija uspoređuje **10 jezičnih modela** iz 2 providera (Ollama lokalno, Groq cloud) na skupu od **50 evaluacijskih pitanja** pokrivajući 6 tipova upita. Treći provider (Gemini) je pokušan, ali isključen iz rezultata — vidi napomenu u poglavlju "Testirani modeli".

> **Napomena o ovoj reviziji:** Ovaj izvještaj je u potpunosti ponovno generiran iz tri benchmark runa (`benchmark_20260621_134732`, `benchmark_20260621_165819`, `benchmark_20260621_185454`) nakon što je otkriveno i ispravljeno da je raniji sudac (`llama3.2`) istovremeno bio i testirani kandidat (self-judging bias), te da su tri Groq modela u ranijem runu koristila zastarjele ID-ove koji su na Groq API-ju vraćali `404 model_not_found`. Sve brojke u ovom dokumentu izračunate su izravno iz sirovih JSON rezultata, isključujući neuspjele/pogrešne zapise — vidi poglavlje "Datoteke rezultata" za potpunu sljedivost.

---

## Metodologija

### Skup evaluacijskih pitanja

Evaluacijski skup (`data/eval_questions.json`) sadrži **50 pitanja** raspoređenih po:

**6 tipova upita:**

| Tip upita | Broj pitanja | Opis | Primjer |
|---|---|---|---|
| `topic_search` | 9 | Pronalazak radova o temi | "What papers exist about graph neural networks?" |
| `method_usage` | 9 | Pronalazak primjena metode | "Which papers use attention mechanisms for NLP tasks?" |
| `method_comparison` | 8 | Usporedba metoda | "Which papers compare CNN and RNN for text classification?" |
| `dataset_search` | 8 | Pronalazak radova koji koriste skup podataka | "Which papers use the ImageNet dataset?" |
| `claim_verification` | 8 | Provjera tvrdnje | "Does BERT outperform traditional methods on question answering?" |
| `paper_lookup` | 8 | Pronalazak specifičnog rada | "Find the paper titled 'Attention Is All You Need'." |

**3 razine težine:** `easy` (14 pitanja), `medium` (20 pitanja), `hard` (16 pitanja)

Svako pitanje sadrži `answer_hints` — ključne pojmove koje dobar odgovor treba sadržavati (npr. za "Attention Is All You Need": `["Vaswani", "multi-head attention", "encoder-decoder", "2017"]`).

### Pipeline

Za svako pitanje pokreće se cijeli **9-stage RAG pipeline**:

```
pitanje → klasifikacija tipa upita → ekstrakcija entiteta → normalizacija sinonima
       → paralelni SPARQL upiti prema ORKG-u → rangiranje → filtriranje
       → izgradnja konteksta + formatiranje izvora → LLM generacija
```

Svaki model koristi se za sve tri LLM faze (klasifikacija, ekstrakcija, generacija). SPARQL upiti prema ORKG-u su deterministički — isti za sve modele.

### LLM-as-Judge evaluacija

Svaki generirani odgovor ocjenjuje neovisni model-sudac na **3 metrike** skale 0–5:

| Metrika | Opis |
|---|---|
| **Groundedness** | Je li odgovor temeljen isključivo na dohvaćenim izvorima iz ORKG-a? |
| **Relevance** | Odgovara li odgovor na postavljeno pitanje? |
| **Completeness** | Pokriva li odgovor ključne aspekte navedene u `answer_hints`? |

**Overall score** = aritmetička sredina svih triju metrika.

Sudac prima: pitanje, generirani odgovor, naslove dohvaćenih izvora i `answer_hints`. Odgovara isključivo JSON-om bez slobodnog teksta.

**Sudac korišten u ovom izvještaju: `ollama:qwen3:8b`.** Ovaj model nije kandidat u glavnoj usporedbi (vidi tablicu "Testirani modeli"), čime se izbjegava self-judging bias. `eval/benchmark.py` od ove revizije odbija pokretanje (hard error) ako se `--judge` poklapa s nekim modelom iz `--models`; `eval/rejudge.py` u istom slučaju ispisuje upozorenje. Ranija verzija ovog izvještaja koristila je `llama3.2` kao suca, koji je istovremeno bio i testirani kandidat — taj problem je ovom revizijom ispravljen, a stare brojke su u potpunosti zamijenjene.

Tehnička napomena: `qwen3:8b` po defaultu koristi "thinking" način rada (`<think>...</think>` razmišljanje prije odgovora), koji je u testiranju trošio cijeli token budžet bez generiranja stvarnog JSON odgovora. Riješeno dodavanjem `"think": false` parametra u `OllamaClient` (`backend/llm/ollama_client.py`) — generička izmjena koja je no-op za modele bez thinking moda.

### Tehničke pojedinosti

- **Benchmark skripta:** `eval/benchmark.py` — CLI s parametrima `--models`, `--judge`, `--delay`, `--judge-delay`, `--limit`, `--query-types`, `--skip-judge`
- **Rejudge skripta:** `eval/rejudge.py` — retroaktivno bodovanje postojećih rezultata novim sucem
- **Izlaz:** JSON s potpunim per-pitanje rezultatima + CSV sažetak po modelu
- **Delay između pitanja:** 16s za sve modele (postavljeno konzervativno zbog Gemini/Groq free-tier rate limita; lokalni Ollama modeli ne trebaju delay, ali su pokretani s istom vrijednosti jer CLI primjenjuje jedinstveni `--delay` na cijeli run)
- **Delay između judge pozivâ:** 1s
- **SPARQL timeout:** 10s po upitu

---

## Testirani modeli

| Provider | Model (stvarni API ID) | Veličina | Tip |
|---|---|---|---|
| Ollama (lokalno) | `llama3` | 8B | Meta Llama 3 |
| Ollama (lokalno) | `llama3.2` | 3B | Meta Llama 3.2 |
| Ollama (lokalno) | `mistral` | 7B | Mistral AI |
| Ollama (lokalno) | `phi3:mini` | 3.8B | Microsoft Phi-3 |
| Ollama (lokalno) | `qwen2.5:7b` | 7B | Alibaba Qwen 2.5 |
| Groq API | `llama-3.1-8b-instant` | 8B | Meta Llama 3.1 |
| Groq API | `llama-3.3-70b-versatile` | 70B | Meta Llama 3.3 |
| Groq API | `openai/gpt-oss-20b` | 20B | OpenAI OSS |
| Groq API | `meta-llama/llama-4-scout-17b-16e-instruct` | 17B aktivnih (MoE) | Meta Llama 4 Scout |
| Groq API | `qwen/qwen3-32b` | 32B | Alibaba Qwen 3 |

Svaki model evaluiran na svih **50 pitanja** (ukupno 500 pipeline izvođenja).

**Napomena o Groq model ID-ovima:** tri od pet Groq modela (`gpt-oss-20b`, `llama-4-scout-17b-16e-instruct`, `qwen3-32b`) zahtijevaju namespace prefiks (`openai/`, `meta-llama/`, `qwen/`) — stari ID-ovi bez prefiksa vraćaju `404 model_not_found`. Ovo je otkriveno tijekom ovog runa; tablica gore navodi ispravne, trenutno važeće ID-ove (provjereno preko `GET /v1/models` na Groq API-ju).

**Isključen model:** `gemini:gemini-2.5-flash-lite` je pokušan u dva navrata, ali isključen iz rezultata. Prvi pokušaj je vraćao `503 UNAVAILABLE` (privremeno preopterećenje servisa) na 44/50 pitanja; drugi pokušaj je nakon par uspješnih poziva potrošio **dnevnu free-tier kvotu** (20 requesta/dan po modelu) i vraćao `429 RESOURCE_EXHAUSTED`. Niti jedan pokušaj nije dao upotrebljiv skup od 50 validnih odgovora, pa je model izostavljen iz analize. Ovo je ograničenje Gemini free-tier kvote, ne greška u pipelineu (vidi poglavlje "Ograničenja evaluacije").

---

## Rezultati

### Opći poredak (sortirano po Overall score)

| # | Model | Overall ↓ | Groundedness | Relevance | Completeness | Latency | n |
|---|---|---|---|---|---|---|---|
| 1 | `groq:gpt-oss-20b` | **4.226** | 4.460 | 4.740 | 3.480 | 3.3s | 50/50 |
| 2 | `groq:qwen3-32b` | **3.731** | 3.809 | 3.745 | 3.638 | 11.3s | 50/50 |
| 3 | `groq:llama-3.3-70b-versatile` | **3.618** | 3.735 | 4.245 | 2.878 | 5.3s | 49/50¹ |
| 4 | `ollama:qwen2.5:7b` | **3.585** | 3.700 | 4.200 | 2.860 | 13.0s | 50/50 |
| 5 | `groq:llama-4-scout-17b` | **3.566** | 3.620 | 4.260 | 2.820 | 4.9s | 50/50 |
| 6 | `groq:llama-3.1-8b-instant` | **3.459** | 3.460 | 4.120 | 2.800 | 4.3s | 50/50 |
| 7 | `ollama:phi3:mini` | **3.432** | 3.520 | 3.920 | 2.860 | 13.0s | 50/50 |
| 8 | `ollama:llama3` | **3.413** | 3.440 | 4.080 | 2.720 | 12.5s | 50/50 |
| 9 | `ollama:mistral` | **3.147** | 3.020 | 3.740 | 2.680 | 15.3s | 50/50 |
| 10 | `ollama:llama3.2` | **3.085** | 2.960 | 3.700 | 2.600 | 9.0s | 50/50 |

¹ Jedno pitanje (Q50, `paper_lookup`) na ovom modelu je naišlo na prolazni `429 rate limit` na Groq API-ju tijekom generacije odgovora; taj zapis je isključen iz prosjeka kvalitete (judge bi ga inače ocijenio 0/0/0, što bi nepravedno snizilo prosjek za infrastrukturni, ne modelski, propust).

### Rezultati po tipu upita (prosjek svih 10 modela)

| Tip upita | Prosječni Overall | n |
|---|---|---|
| `topic_search` | **3.936** | 90 |
| `method_usage` | 3.620 | 90 |
| `claim_verification` | 3.565 | 80 |
| `method_comparison` | 3.541 | 80 |
| `dataset_search` | 3.308 | 80 |
| `paper_lookup` | 3.117 | 79¹ |

¹ Q50 (paper_lookup) na `llama-3.3-70b-versatile` isključen — vidi napomenu uz Opći poredak.

### Rezultati po tipu upita, po modelu

| Tip upita | gpt-oss-20b | qwen3-32b | llama-3.3-70b | qwen2.5:7b | llama-4-scout | llama-3.1-8b | phi3:mini | llama3 | mistral | llama3.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `topic_search` | 4.63 | 4.25 | 3.96 | 4.11 | 3.96 | 4.00 | 3.44 | 3.52 | 4.04 | 3.48 |
| `method_usage` | 4.37 | 4.62 | 3.55 | 3.33 | 3.78 | 3.37 | 3.52 | 3.70 | 3.18 | 3.00 |
| `claim_verification` | 4.12 | 1.54 | 3.96 | 4.33 | 3.87 | 3.67 | 3.75 | 3.75 | 3.50 | 3.17 |
| `method_comparison` | 4.17 | 3.33 | 3.71 | 3.46 | 3.87 | 3.12 | 3.87 | 3.75 | 2.79 | 3.33 |
| `dataset_search` | 3.92 | 5.00 | 2.92 | 3.25 | 3.04 | 3.21 | 2.71 | 3.13 | 3.00 | 2.92 |
| `paper_lookup` | 4.08 | 3.75 | 3.57 | 3.00 | 2.79 | 3.33 | 3.29 | 2.58 | 2.25 | 2.58 |

Napomena: `qwen3-32b` na `dataset_search` postiže savršenih 5.00 — uzorak za taj presjek je malen (8 pitanja), pa je ova brojka osjetljiva na varijancu pojedinačnih odgovora, ne nužno reprezentativna za opću sposobnost modela na tom tipu upita. Isto vrijedi i za `qwen3-32b` na `claim_verification` (1.54) — drastično niže od svih ostalih modela na istom presjeku, vrijedi provjeriti pojedinačne odgovore prije izvođenja zaključaka u diskusiji.

### Rezultati po težini pitanja (prosjek svih 10 modela)

| Težina | Prosječni Overall | n |
|---|---|---|
| `easy` | **3.676** | 140 |
| `hard` | 3.572 | 159 |
| `medium` | 3.383 | 200 |

---

## Ablation studija: utjecaj normalizacije entiteta i hard filtera

Za razliku od modelske usporedbe iznad, ova studija izolira utjecaj dvaju pipeline koraka — **normalizacije entiteta** (Step 3, rječnik sinonima) i **hard filtera** (Step 6, zadržava papire koji imaju i podudaranje metode i skupa podataka kad su oba entiteta prisutna) — na retrieval, neovisno o kvaliteti LLM-generiranog odgovora.

### Metodologija

- Pokrenuto preko `eval/ablation.py` na svih **50 pitanja**, 4 konfiguracije (normalizacija on/off × hard filter on/off), fiksni model `ollama:llama3` (besplatno, lokalno).
- Bez LLM-as-judge bodovanja — mjere se samo retrieval-level metrike: broj sirovih KG rezultata (`kg_results_count`), broj izvora nakon truncationa (`sources_count`) i latencija.
- `RAGPipeline` podržava `enable_normalization` i `enable_hard_filter` flagove upravo za ovu svrhu (vidi `backend/rag/pipeline.py`).

### Rezultati

| Konfiguracija | Prosj. KG rezultati | Prosj. sources | Prosj. latencija | Neuspjelih |
|---|---|---|---|---|
| normalizacija ON, filter ON | **29.400** | 6.800 | 13.031s | 0/50 |
| normalizacija OFF, filter ON | 20.240 | 6.380 | 12.566s | 0/50 |
| normalizacija ON, filter OFF | 29.200 | 7.340 | 12.686s | 0/50 |
| normalizacija OFF, filter OFF | 20.440 | 6.920 | 11.961s | 0/50 |

*(Izvor: `eval/results/ablation_20260621_133709.json` / `_summary.csv`)*

### Analiza

**Normalizacija entiteta povećava recall za ~44%.** Uz uključenu normalizaciju (sinonimi/varijante za METHOD/DATASET/TASK/FIELD), pipeline dohvaća prosječno 29.3 KG rezultata po pitanju, naspram 20.3 bez nje — razlika je konzistentna kod oba stanja hard filtera. To potvrđuje da rječnik sinonima (117 method + 62 dataset + 28 task + 11 field varijanti) stvarno proširuje SPARQL upite na pojmove koje korisnik ne navodi izravno (npr. "CNN" → "convolutional neural network"), umjesto da bude kozmetička značajka.

**Hard filter mjerljivo, ali umjereno smanjuje broj zadržanih izvora (~7–8%).** Uz normalizaciju ON, filter smanjuje prosječan broj sources s 7.34 na 6.80; uz normalizaciju OFF, sa 6.92 na 6.38. Učinak je manji nego što bi se moglo očekivati zato što hard filter samo djeluje na pitanja koja imaju **istovremeno** ekstrahiran method i dataset entitet — na evaluacijskom skupu to je samo **14/50 pitanja (28%)**. Na preostalih 72% pitanja (samo method, samo dataset, ili niti jedan) filter je no-op po definiciji (`hard_filter` u `backend/rag/ranking.py` ne filtrira ako nedostaje jedna od dvije kategorije). Filter dakle radi kako je projektirano — efekt na ukupni prosjek je razrijeđen sastavom evaluacijskog skupa, ne bug u implementaciji.

**Latencija nije osjetljivo različita** između konfiguracija (11.96–13.03s) — normalizacija i filter su jeftine operacije na već dohvaćenim podacima (in-memory), dominantni trošak je SPARQL mreža + LLM generacija, ne ove dvije faze.

---

## Analiza i zaključci (modelska usporedba)

### 1. Groq cloud modeli dominiraju ljestvicom, ali ne isključivo zbog veličine

Četiri od pet najboljih modela su Groq cloud modeli (`gpt-oss-20b`, `qwen3-32b`, `llama-3.3-70b-versatile`, `llama-4-scout-17b`), ali ne strogo po veličini — `gpt-oss-20b` (20B) nadmašuje `llama-3.3-70b-versatile` (70B) i `qwen3-32b` (32B) za znatnu razliku (4.226 vs 3.618 / 3.731). Veličina modela ne predviđa pouzdano kvalitetu odgovora u ovom zadatku.

### 2. Lokalni (Ollama) modeli zaostaju, ali ne dramatično

Najbolji lokalni model (`qwen2.5:7b`, Overall 3.585) zaostaje za najboljim cloud modelom (`gpt-oss-20b`, 4.226) za ~15%, ali nadmašuje tri Groq modela (`llama-4-scout-17b`, `llama-3.1-8b-instant`) te je usporediv s `llama-3.3-70b-versatile`. Za besplatan, offline deployment, `qwen2.5:7b` je trenutno najbolji izbor u testiranom skupu.

### 3. Paper lookup i dataset search su najslabiji tipovi upita

Za razliku od uobičajenog očekivanja da je `claim_verification` najteži (jer traži empirijsku provjeru tvrdnje), u ovoj reviziji `paper_lookup` (3.117) i `dataset_search` (3.308) su najslabiji. `paper_lookup` traži pronalazak točno određenog rada po naslovu — niski score sugerira da naslov-temeljeni fallback retrieval (title matching) nije dovoljno precizan kad LLM ne formulira naslov identično ORKG zapisu.

### 4. `qwen3-32b` ima ekstremnu varijancu po tipu upita

`qwen3-32b` postiže najviši score na `dataset_search` (5.00) i `method_usage` (4.62), ali drastično najniži na `claim_verification` (1.54) — daleko ispod svih ostalih modela na tom presjeku (sljedeći najniži je `llama3.2` s 3.17). Ovo zaslužuje provjeru pojedinačnih odgovora (`eval/results/benchmark_20260621_185454.json`) prije nego se generalizira kao svojstvo modela — moguće je da je riječ o specifičnom modu kvara (npr. model odbija odgovoriti na pitanja tipa "Does X outperform Y" zbog svoje sigurnosne politike).

### 5. Relevance > Groundedness > Completeness kod većine modela

Slično ranijem (kontaminiranom) izvještaju, modeli generalno bolje "odgovaraju na pitanje" (Relevance) nego što potpuno pokrivaju očekivane aspekte (Completeness) — uzorak sugerira da je ORKG pokrivenost (koliko aspekata pitanja knowledge graph stvarno sadrži), a ne sposobnost modela, glavni ograničavajući faktor za Completeness.

---

## Ograničenja evaluacije

1. **LLM-as-judge pristranost** — sudac (`qwen3:8b`) ima vlastite predrasude i može favorizirati određeni stil odgovora. Self-judging bias iz ranije verzije ovog izvještaja (sudac `llama3.2` je istovremeno bio i kandidat) je ispravljen — sudac u ovoj reviziji nije kandidat u glavnoj usporedbi. Idealno bi i dalje bio korišten jači model (GPT-4o, Claude Opus) kao sudac za apsolutnu kalibraciju, no to nije bilo dostupno (nema OpenAI/Anthropic API key-a u ovom okruženju).

2. **Gemini nije evaluiran** — free-tier kvota (20 requesta/dan po modelu) iscrpljena je tijekom dva pokušaja, pa `gemini-2.5-flash-lite` nije dao upotrebljiv skup rezultata. Usporedba je ograničena na Ollama i Groq.

3. **Answer hints su nužni uvjet, ne dovoljan** — `answer_hints` provjera ne mjeri je li odgovor faktično točan, nego sadrži li ključne pojmove. Model može pogoditi ključne riječi bez razumijevanja.

4. **ORKG pokrivenost nije kontrolirana** — neka pitanja mogu biti lakša jer ORKG slučajno ima više podataka o toj temi. Evaluacijski skup nije stratificiran po ORKG pokrivenosti.

5. **Jedan sudac za sve modele** — konzistentnost je osigurana, ali apsolutne vrijednosti scoreva ovise o kalibraciji suca.

6. **Mali uzorak po presjeku (tip upita × model)** — 8–9 pitanja po tipu upita znači da pojedinačni ekstremni rezultati (npr. `qwen3-32b` na `claim_verification`) mogu znatno pomaknuti prosjek. Ove brojke treba tumačiti kao indikativne, ne statistički robusne na razini presjeka.

7. **Jedan prolazni infrastrukturni propust** — Q50 na `llama-3.3-70b-versatile` je naišao na `429 rate limit` kod Groq API-ja; taj zapis je isključen iz prosjeka (model je evaluiran na 49/50 umjesto 50/50 pitanja za tu jednu metriku).

---

## Datoteke rezultata

| Datoteka | Sadržaj | Validni modeli korišteni u ovom izvještaju |
|---|---|---|
| `eval/results/benchmark_20260621_134732.json` | 7 modela (judge: `qwen3:8b`) | `ollama:llama3`, `ollama:llama3.2`, `ollama:mistral`, `ollama:phi3:mini`, `ollama:qwen2.5:7b`, `groq:llama-3.1-8b-instant` — **isključen:** `gemini:gemini-2.5-flash-lite` (503 overload na 44/50) |
| `eval/results/benchmark_20260621_165819.json` | 5 modela (judge: `qwen3:8b`) | `groq:llama-3.3-70b-versatile` — **isključeni:** `gemini:gemini-2.5-flash-lite` (429 kvota), `groq:gpt-oss-20b`/`groq:llama-4-scout-17b-16e-instruct`/`groq:qwen3-32b` (404, zastarjeli ID-ovi bez namespace prefiksa) |
| `eval/results/benchmark_20260621_185454.json` | 3 modela, ispravljeni ID-ovi (judge: `qwen3:8b`) | `groq:openai/gpt-oss-20b`, `groq:meta-llama/llama-4-scout-17b-16e-instruct`, `groq:qwen/qwen3-32b` |
| `eval/results/ablation_20260621_133709.json` | Ablation studija (50 pitanja, 4 konfiguracije norm×filter, `ollama:llama3`) | — (retrieval-level, bez judge bodovanja) |

Stariji rezultati (`benchmark_20260413_*`, `benchmark_20260421_*`) korišteni su u prijašnjoj, **povučenoj** verziji ovog izvještaja koja je sadržavala self-judging bias (sudac `llama3.2` istovremeno kandidat) i nisu korišteni za brojke u ovoj reviziji.

Svaka benchmark JSON datoteka sadrži potpune rezultate po pitanju (pitanje, tip, težina, latencija, odgovor, dohvaćeni izvori, judge scorevi i obrazloženje). Ablation JSON sadrži retrieval-level metrike po konfiguraciji, bez judge bodovanja.
