# Evaluacija dohvata (retrieval) GraphRAG sustava nad ORKG-om

## Sažetak

LLM-as-judge evaluacija (vidi `BENCHMARK_REPORT.md`) pokazala je pristojne ukupne ocjene, ali ručna analiza odgovora otkrila je da ~80 % odgovora na pitanja tipa `method_comparison` i `claim_verification` kaže da u izvorima nema traženih informacija — a sudac takve odgovore nagrađuje (groundedness 5/5 i uz 0 izvora). Da bi se dohvat mjerio neovisno o sucu, izgrađen je **graded gold set relevantnosti** za svih 50 evaluacijskih pitanja, a dohvat je zatim poboljšan **isključivo graf-nativnim tehnikama** (bez vektorskih embeddingsa) i mjeren na odvojenom test skupu.

| | nDCG@8 test (oracle) | nDCG@8 test (end-to-end, qwen2.5:7b) |
|---|---|---|
| Početno stanje (commit `6273927`) | 0,540 | 0,575 |
| **Konačni GraphRAG** | **0,768** | **0,709** |

---

## Gold set

- **Datoteke:** `data/gold/search_spec.json` (pojmovi pretrage i kriterij relevantnosti po pitanju), `data/gold/candidates.json` (skupljeni kandidati s ORKG zapisima), `data/gold/retrieval_gold.json` (ocjene).
- **Ocjene:** 0 = nerelevantno (samo leksičko podudaranje, druga tema), 1 = djelomično relevantno, 2 = izravno relevantno (ORKG zapis sadrži ono što pitanje traži). Kriterij je zapisan za svako pitanje posebno.
- **Pooling (TREC-stil):** kandidati iz pet neovisnih izvora (`eval/build_gold_candidates.py`): trenutni pipeline s ručnim entitetima, claim-evidence upit, pretraga naslova, pretraga vrijednosti kontribucija i lookup točnog naslova. Pool je naknadno proširen svakim radom koji je bilo koja evaluirana verzija sustava dohvatila, tako da **sve prijavljene brojke imaju 0 % neocijenjenih rezultata**.
- **Veličina:** 1896 ocjena (540 × 2, 664 × 1, 692 × 0).
- **Postupak ocjenjivanja:** ocjene je predložio Claude čitajući ORKG zapise prema kriteriju; autor ih je pregledao i odobrio. *(Preporuka za rad: neovisno ocijeniti uzorak i prijaviti Cohenov κ.)*
- **Dev/test podjela:** dev = neparna pitanja, test = parna (25/25, svi tipovi upita u obje). Sve odluke o dohvatu donesene su na dev skupu; test je otvoren tek za konačno mjerenje.

## Metrike

Mjeri se točno onih **K = 8** radova koje LLM dobije u kontekstu (`RAGPipeline.retrieve()`, koraci 1–7), skripta `eval/retrieval_eval.py`:

- **nDCG@8** (glavna metrika) — kvaliteta poretka s ocjenama 2/1/0
- **R@8** — udio izravno relevantnih (2) radova iz gold seta koji su dohvaćeni (gornja granica 0,78, jer neka pitanja imaju i do 34 relevantna rada)
- **P2@8** — udio dohvaćenih radova s ocjenom 2; **Hit@8** — barem jedan rad s ocjenom 2; **empty** — prazan kontekst

Dva načina: **oracle** (točan tip pitanja i ručni entiteti iz `data/eval_questions.json` — izolira dohvat) i **end-to-end** (LLM `qwen2.5:7b` klasificira pitanje i izvlači entitete).

---

## Rezultati

### Verzije sustava

| Verzija | nDCG@8 svih 50 | dev | **test** | R@8 test | P2@8 test | Hit@8 test | prazan kontekst |
|---|---|---|---|---|---|---|---|
| A — početno stanje (`6273927`) | 0,505 | 0,471 | **0,540** | 0,257 | 0,323 | 0,68 | 10 % |
| B — popravak rangiranja + obogaćivanje (`6b36805`) | 0,664 | 0,605 | **0,722** | 0,443 | 0,403 | 0,84 | 2 % |
| C — konačni GraphRAG | 0,724 | 0,680 | **0,768** | 0,435 | 0,428 | 0,84 | 2 % |
| A — end-to-end, qwen2.5:7b | 0,514 | 0,452 | **0,575** | 0,383 | 0,370 | 0,88 | 4 % |
| C — end-to-end, qwen2.5:7b | 0,692 | 0,675 | **0,709** | 0,483 | 0,418 | 0,92 | 0 % |
| C — end-to-end, llama3 | 0,690 | 0,673 | **0,707** | 0,443 | 0,408 | 0,88 | 0 % |

### Po tipu upita (nDCG@8, svih 50)

| Tip upita | A oracle | B oracle | C oracle | A e2e (qwen) | C e2e (qwen) | C e2e (llama3) |
|---|---|---|---|---|---|---|
| `claim_verification` | 0,446 | 0,667 | 0,667 | 0,463 | 0,682 | 0,665 |
| `dataset_search` | 0,803 | 0,844 | 0,844 | 0,844 | 0,833 | 0,833 |
| `method_comparison` | 0,452 | 0,627 | 0,679 | 0,382 | 0,606 | 0,600 |
| `method_usage` | 0,530 | 0,598 | 0,722 | 0,449 | 0,646 | 0,644 |
| `paper_lookup` | 0,000 | 0,474 | 0,583 | 0,221 | 0,544 | 0,549 |
| `topic_search` | 0,765 | 0,766 | 0,834 | 0,708 | 0,829 | 0,835 |

### Ablacija (konačni sustav C, oracle, nDCG@8)

| Konfiguracija | svih 50 | dev | test |
|---|---|---|---|
| **Puni sustav** | **0,724** | **0,680** | **0,768** |
| − dokazi s više pojmova (multi-term) | 0,688 | 0,636 | 0,740 |
| − obogaćivanje (dvostupanjski dohvat) | 0,704 | 0,667 | 0,742 |
| − origin lookup | 0,706 | 0,667 | 0,746 |
| − benchmark traverzija | 0,716 | 0,659 | 0,772 |

Ablacije su mjerene prije pravila klasifikacije (koja na oracle način ne utječu). Pokreću se preko postavki (`ENABLE_MULTI_TERM`, `ENABLE_BENCHMARK`, `ENABLE_ORIGIN`, `ENRICH_CANDIDATES=0`).

---

## Što je promijenjeno i zašto

**Uzroci kvarova u početnom stanju (A):**
1. Rangiranje je bodovalo samo polja `methodLabel` / `datasetLabel`. Redovi iz `claim_evidence` (uloge poput *Baseline*, benchmark rezultati) i iz pretrage naslova dobivali su 0 bodova i soft filter ih je izbacivao — uključujući rad "Attention Is All You Need" koji je pretraga naslova točno pronašla.
2. Hard filter je praznio kontekst kad pitanje ima i metodu i dataset (BERT+GLUE, ResNet+ImageNet, YOLO+COCO).
3. Kontekst je sadržavao samo vrijednost koja se podudarala (`Method: Faster R-CNN`), bez uloge u grafu (`Baseline: Faster R-CNN`, `Superior model: Improved YOLOv7`) i bez rezultata.
4. Izvlačenje naslova iz pitanja zadržavalo je "titled", navodnike i interpunkciju.

**Graf-nativne tehnike (B → C):**

| Tehnika | Graf-struktura koju koristi |
|---|---|
| Rangiranje svjesno uloge | predikat kojim je vrijednost vezana za kontribuciju (method / model / baseline / dataset / benchmark); svaka tražena metoda boduje se posebno |
| Dvostupanjski dohvat (obogaćivanje) | puni zapis kontribucija top-15 kandidata (`Q.papers_details`, jedan `VALUES` upit) + ponovno rangiranje; u kontekst najviše 15 parova po radu |
| Dokazi s više pojmova | kontribucije rangirane po broju različitih pojmova pitanja koje pokrivaju (topic / usage pitanja s više aspekata) |
| Benchmark traverzija | Contribution → `P1004` model, Contribution → `PWC_HAS_BENCHMARK` → Benchmark → `P2005` Dataset / `P34` Evaluation; za usporedbe samo radovi koji pokrivaju ≥ 2 tražene metode |
| Origin lookup | najraniji radovi koji bilježe metodu i broj takvih vrijednosti — rad koji je metodu uveo (word2vec, ResNet, T5) bez podudaranja naslova |
| Lookup po naslovu | navodnici / uklanjanje "titled", "that introduced the"; pretraga naslova po ključnim riječima; točan naslov ima prednost |

**Klasifikacija i oblici entiteta (end-to-end).** Prije LLM klasifikacije pravila odlučuju jasne oblike pitanja (lookup: "find the paper…", "look up…", "the paper that introduced…"; claim: da/ne pitanje s komparativom ili tvrdnjom, bez riječi "papers"; usporedba: "compare", "versus"); ostalo klasificira LLM. Normalizacija svodi oblike koje LLM vrati na oblike iz grafa ("variational autoencoders" → "variational autoencoder", "Vision Transformer (ViT)" → oba oblika). Pravila su na testu pogodila 12/12 pitanja na koja su se primijenila, a točnost tipa za `llama3` porasla je s 19/25 na 22/25 — ali **nDCG@8 na testu nije porastao** (qwen2.5 0,711 → 0,709, llama3 0,716 → 0,707); dobitak od +0,06 vidljiv je samo na dev skupu (qwen2.5 0,616 → 0,675, llama3 0,607 → 0,673). Zadržano jer ne šteti, smanjuje broj LLM poziva i čini klasifikaciju neovisnijom o modelu.

**Isprobano i odbačeno** (mjereno na dev skupu): IDF težina pojmova prema broju radova u grafu (raspon 0,5–1,5: dev 0,685 → 0,676; 0,75–1,25: 0,679); težina taska +2; 15 → 30 kandidata za obogaćivanje; bodovanje research fielda (i u verziji ograničenoj na research problem čvorove); field kao research problem upit; ključne fraze iz pitanja u claim evidence; filtar benchmark traverzije po research problemu; semantičko rerankiranje embeddingsima (+0,04 nDCG na dev skupu, uklonjeno jer sustav namjerno ostaje čisti GraphRAG).

## Ograničenja i iskreni nalazi

- **Benchmark traverzija gotovo ne utječe na metrike dohvata**: doprinosi +0,009 nDCG na svih 50 pitanja, a na test skupu je bez nje rezultat čak 0,004 bolji. Njena svrha je drugačija: u kontekst donosi stvarne brojke (npr. `Coco result: Box ap_50.8 | Fps_126`), a to mjeri tek kvaliteta LLM odgovora, ne metrike dohvata.
- Na test skupu C je kod `claim_verification` i `method_comparison` na razini B, a ne bolji; dobici C su u `method_usage`, `paper_lookup` i `topic_search`.
- Mali skup (50 pitanja; po tipu upita 8–9), jedan graf, jedan oracle skup entiteta.
- Radovi izvan poola tretiraju se kao nerelevantni; pool je proširen sa svim dohvaćenim radovima evaluiranih verzija, ali ne sa svim mogućim radovima u ORKG-u.
- Ocjene je predložio LLM i pregledao autor — nedostaje neovisna mjera slaganja ocjenjivača.
- Lookup ostaje najslabiji kad naslov nema zajedničkih riječi s pitanjem, a ORKG zapis ne bilježi godinu (ELMo).

## Datoteke rezultata

| Datoteka | Sadržaj |
|---|---|
| `eval/results/retrieval_oracle_final.json` | konačni sustav C, oracle, svih 50 pitanja |
| `eval/results/retrieval_e2e_{qwen2.5_7b,llama3}_final.json` | konačni sustav C, end-to-end |
| `eval/results/retrieval_e2e_{qwen2.5_7b,llama3}_before_rules.json` | C prije pravila klasifikacije i normalizacije oblika |
| `eval/results/retrieval_ablation_*.json` | ablacije (oracle, svih 50) |

Verzije A i B mjerene su iz git worktreeja na commitima `6273927` i `6b36805` s istim konačnim gold setom.
