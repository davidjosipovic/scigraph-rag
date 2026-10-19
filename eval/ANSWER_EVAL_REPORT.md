# Evaluacija odgovora GraphRAG sustava: gold-based metrike i faithfulness

## Sažetak

`BENCHMARK_REPORT.md` je kvalitetu odgovora mjerio jednim LLM sucem: ocjene 0–5 za groundedness, relevance i completeness. Taj sudac vidi samo naslove izvora, ne i njihov sadržaj. Zato ne može provjeriti je li ono što odgovor tvrdi stvarno u grafu, niti zna koji su izvori relevantni. Ovaj izvještaj tog suca zamjenjuje s dvije metrike:

1. **Gold Score** je glavna metrika i ne koristi LLM. Radovi koje odgovor citira čitaju se deterministički, preko točnog naslova. Zatim se uspoređuju s graded gold setom relevantnosti (`data/gold/retrieval_gold.json`) i s presudama za `claim_verification` pitanja (`data/gold/claim_verdicts.json`).
2. **Faithfulness** je sekundarna metrika. Lokalni sudac (`ollama:qwen3:8b`) rastavlja odgovor na pojedinačne tvrdnje. Svaku tvrdnju provjerava prema cijelom kontekstu koji je model vidio i mora doslovno citirati redak kojim je potvrđuje. Sudac je validiran na 100 tvrdnji s referentnim oznakama (κ = 0,51).

Glavni rezultati na 9 modela i 50 pitanja, nakon poboljšanja dohvata (`RETRIEVAL_REPORT.md`):

| | Gold Score | Faithful |
|---|---|---|
| Najbolji cloud model (`gpt-oss-20b`) | **0,830** | 0,694 |
| Najbolji lokalni model (`qwen2.5:7b`) | **0,758** | 0,771 |
| Najvjernija kontekstu (`qwen3.8-27b`) | 0,790 | **0,814** |

Na istim odgovorima stari sudac daje cloud modelima 4,3–4,6, a lokalnima 3,3–3,5 (od 5). To je razlika od oko 22 % skale. Gold Score između najboljeg cloud i najboljeg lokalnog modela razlikuje se za 7 postotnih bodova. Unutar lokalnih modela sudac gotovo ne razlikuje kvalitetu: Spearman ρ s gold Scoreom je 0,25.

---

## Zašto nova metrika

Ručni pregled odgovora pokazao je tri problema starog suca:

- **Ne vidi sadržaj konteksta.** Dobiva samo naslove izvora, pa ne može znati je li npr. tvrdnja "rad X trenira CNN-RNN baseline" u grafu ili ju je model izmislio.
- **Nema gold standarda.** Ne zna koji su dohvaćeni radovi relevantni, pa citiranje nerelevantnog rada ne kažnjava.
- **Model-sudac je sam LLM.** Njegove ocjene ovise o stilu odgovora. Mjereno na ovom runu, slaže se s gold Scoreom samo umjereno (vidi "Stari sudac prema novim metrikama").

Sudac mora biti lokalan. Groq i Gemini su na free tieru, s ograničenjem broja zahtjeva i bez jamstva da će model ostati dostupan. Dva Groq modela iz ranijeg izvještaja (`llama-3.3-70b`, `llama-4-scout`) u međuvremenu su povučena. Zato je sudac `qwen3:8b` preko Ollame, a sudac nije nijedan od testiranih modela.

---

## Metodologija

### Gold Score (`eval/answer_eval.py`)

Svaki odgovor slijedi predložak sa sekcijama Overview, Relevant Papers, Methods, Datasets, Sources i Verdict/Evidence za claim pitanja, a radove navodi točnim naslovom. Za svaki odgovor računa se sljedeće:

| Veličina | Definicija |
|---|---|
| *citirano* | izvori iz konteksta čiji se naslov pojavljuje u tijelu odgovora; završna lista Sources se ignorira jer modeli u nju stavljaju sve izvore |
| *answerable* | kontekst sadrži barem jedan rad ocjene 2 |
| *abstain* | odgovor ne citira nijedan rad ili za claim kaže INSUFFICIENT |
| **CitePrec** | prosječna gold ocjena citiranih radova / 2 (graded preciznost) |
| **CiteUtil** | udio radova ocjene 2 iz konteksta koje je odgovor citirao |
| **FalseAbst** | udio *answerable* pitanja na koja je model odbio odgovoriti |
| **CorrAbst** | udio pitanja bez odgovora na koja je model ispravno odbio odgovoriti |
| **Fabr** | udio odgovora čija lista Sources sadrži naslov kojeg nema među dohvaćenim izvorima |
| **Verdict** | točnost presude na `claim_verification` pitanjima (7 pitanja; Q20 je MIXED i izuzet) |

**Score po pitanju** se računa ovisno o vrsti pitanja:
- Ako se na pitanje može odgovoriti, Score je F1(CitePrec, CiteUtil), a 0 ako model odbije odgovoriti.
- Ako se ne može, Score je 1 kad model odbije odgovoriti. Inače je jednak CitePrec, što je najviše 0,5 jer nijedan citirani rad nema ocjenu 2.
- Claim pitanja s gold presudom boduju se presudom: 1 ako je prihvatljiva, inače 0. INSUFFICIENT se prihvaća i kad nijedan od ključnih radova nije dohvaćen.

Gold presude za claim pitanja predložio je Claude na temelju ORKG zapisa, a autor ih je pregledao i odobrio 19. 10. 2026. Radove koje je novi dohvat donio, a nisu bili u gold setu (129 radova), ocijenio je Claude prema kriterijima pojedinog pitanja. Ti radovi su označeni s `found_by: benchmark_20261019`. Nakon toga nijedan citirani rad nije ostao neocijenjen.

### Faithfulness (`eval/faithfulness.py`)

1. **Izvlačenje tvrdnji.** Sudac rastavlja tijelo odgovora (bez liste Sources) na najviše 25 samostalnih tvrdnji. Svaka je označena kao *fact* ili *absence* (tvrdnja da nešto ne postoji).
2. **Provjera.** Svaka tvrdnja provjerava se zasebnim pozivom prema cijelom kontekstu. Oznake su S (podržano), U (nije u kontekstu) i C (kontekst kaže drugačije).
3. **Provjera citata.** Za S i C sudac mora doslovno prepisati odlučujući redak. Ako tog retka nema u kontekstu, oznaka postaje U. Iznimka je tvrdnja o odsutnosti označena kao S, za koju citat nije potreban.

Faithful je udio tvrdnji označenih S po odgovoru, uprosječen po modelu. Ocijenjeno je svih 450 odgovora (7 694 tvrdnje). Kontekst je stavljen na početak prompta, pa Ollama može ponovno koristiti prefiks za sve tvrdnje istog odgovora.

### Validacija suca

Uzorak čini 100 nasumičnih tvrdnji, 50 iz lokalnih i 50 iz Groq odgovora (`faithfulness_benchmark_20261019_*_sample.json`). Claude ih je označio bez uvida u oznake suca. Autor je pregledao svih 21 neslaganja i prihvatio Claudeove oznake (`faithfulness_20261019_review.md`). To je validacija jačim modelom uz ljudski pregled neslaganja, a ne neovisna ljudska anotacija svih 100 tvrdnji.

| | n | Točnost | Cohenov κ |
|---|---|---|---|
| S / U / C | 100 | 0,79 | **0,51** |
| Podržano / nije podržano | 100 | 0,84 | **0,60** |

Matrica zabune (retci su referentne oznake, stupci oznake suca):

```
         S    U    C
    S   64    7    0
    U    7   14    0
    C    2    5    1
```

- Sudac griješi u oba smjera podjednako, 7 puta prestrogo i 7 puta preblago. Udio S u uzorku zato je gotovo isti (sudac 73 %, referenca 71 %), pa prosjeci po modelu nisu pristrani. Pojedinačne oznake su umjereno pouzdane.
- Oznaku C sudac gotovo ne koristi. Kad tvrdnja podatak jednog rada pripiše drugom radu, sudac to označi kao U ("nema dokaza"), a ne kao C. **U i C se zato izvještavaju zajedno kao *nepodržano*.**
- Ostali tipični promašaji su:
  - dodani detalj uz pojam koji jest u kontekstu prolazi kao S
  - kod zapisa s više modela sudac ne zna kojem modelu pripada koji rezultat (ORKG redak `X result` stoji iznad svog `model:` retka)
  - točna tvrdnja padne na provjeri citata jer je sudac spojio dva retka

### Runovi

| Run | Modeli | Napomena |
|---|---|---|
| `benchmark_20261019_105454` | llama3, llama3.2, mistral, phi3:mini, qwen2.5:7b (Ollama) | `num_ctx` 8192 |
| `benchmark_20261019_121003` | gpt-oss-20b, gpt-oss-120b, qwen3.8-27b, allam-2-7b (Groq) | 30 s pauze zbog limita |

Kontekst koji je model vidio sprema se uz svaki odgovor (`QuestionResult.context`).

**Popravljena greška:** Ollama je po defaultu koristila `num_ctx` od 4 096 tokena, što ovisi o VRAM-u. Duži promptovi su se tiho rezali s početka, pa je dio konteksta nestajao. Sada je `num_ctx` eksplicitno 8 192 (`backend/config.py`). To je utjecalo na lokalne runove od 16. 10. (npr. Q37, Q50).

---

## Rezultati

### Opći poredak (19. 10. 2026, 50 pitanja)

| # | Model | Score ↓ | Score test | CitePrec | CiteUtil | FalseAbst | Fabr | Verdict | Faithful | Nepodrž. | Stari sudac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `groq:gpt-oss-20b` | **0,830** | 0,833 | 0,832 | 0,883 | 0,03 | 0,00 | 0,86 | 0,694 | 0,278 | 4,57 |
| 2 | `groq:gpt-oss-120b` | **0,798** | 0,745 | 0,793 | 0,900 | 0,03 | 0,00 | 0,86 | 0,722 | 0,246 | 4,53 |
| 3 | `groq:qwen3.8-27b` | **0,790** | 0,782 | 0,724 | 0,983 | 0,03 | 0,00 | 0,86 | **0,814** | 0,183 | 4,31 |
| 4 | `ollama:qwen2.5:7b` | **0,758** | 0,779 | 0,765 | 0,862 | 0,02 | 0,02 | 0,71 | 0,771 | 0,218 | 3,46 |
| 5 | `ollama:mistral` | **0,680** | 0,706 | 0,723 | 0,800 | 0,03 | 0,06 | 0,57 | 0,704 | 0,292 | 3,41 |
| 6 | `ollama:llama3` | **0,669** | 0,705 | 0,730 | 0,749 | 0,00 | 0,00 | 0,57 | 0,684 | 0,315 | 3,27 |
| 7 | `ollama:phi3:mini` | **0,663** | 0,656 | 0,674 | 0,804 | 0,07 | 0,12 | 0,71 | 0,703 | 0,284 | 3,49 |
| 8 | `ollama:llama3.2` | **0,648** | 0,676 | 0,615 | 0,855 | 0,00 | 0,02 | 0,57 | 0,668 | 0,332 | 3,39 |
| 9 | `groq:allam-2-7b` | **0,167** | 0,180 | 0,614 | 0,108 | 0,78 | 0,32 | 0,00 | 0,195 | 0,846 | 1,17 |

*Score test* je prosjek na 25 test pitanja (parni ID-ovi), koja nisu korištena pri razvoju dohvata. *Nepodrž.* je udio tvrdnji označenih U ili C. *Stari sudac* je Overall (0–5) suca iz `BENCHMARK_REPORT.md` na istim odgovorima.

### Po tipu upita (prosjek 8 modela, bez allam-2-7b)

| Tip upita | Score | CitePrec | CiteUtil | Stari sudac |
|---|---|---|---|---|
| `topic_search` | **0,895** | 0,880 | 0,953 | 4,07 |
| `dataset_search` | 0,808 | 0,837 | 0,812 | 3,62 |
| `method_usage` | 0,736 | 0,699 | 0,882 | 3,72 |
| `paper_lookup` | 0,672 | 0,608 | 0,980 | 3,82 |
| `claim_verification` | 0,645 | 0,682 | 0,625 | 4,07 |
| `method_comparison` | **0,599** | 0,660 | 0,660 | 3,49 |

### Učinak dohvata: 16. 10. prema 19. 10.

Između dva runa promijenjeni su dohvat (graf-nativne strategije, pravila za klasifikaciju upita i površinski oblici entiteta, vidi `RETRIEVAL_REPORT.md`) i `num_ctx`. Promjena `num_ctx` utječe samo na lokalne modele. Score za oba runa računat je s istim, sadašnjim gold setom.

| Model | Answerable 16. 10. | Answerable 19. 10. | Score 16. 10. | Score 19. 10. | Δ |
|---|---|---|---|---|---|
| `llama3` | 29 | 40 | 0,484 | 0,669 | +0,185 |
| `mistral` | 30 | 40 | 0,535 | 0,680 | +0,145 |
| `llama3.2` | 33 | 38 | 0,574 | 0,648 | +0,074 |
| `qwen2.5:7b` | 31 | 41 | 0,673 | 0,758 | +0,085 |
| `phi3:mini` | 32 | 41 | 0,652 | 0,663 | +0,011 |
| `gpt-oss-20b` | 31 | 38 | 0,779 | 0,830 | +0,051 |
| `gpt-oss-120b` | 31 | 40 | 0,700 | 0,798 | +0,098 |
| `qwen3.8-27b` | 31 | 39 | 0,681 | 0,790 | +0,109 |
| `allam-2-7b` | 31 | 37 | 0,288 | 0,167 | −0,121 |

Na Groq modele promjena `num_ctx` ne utječe, pa je njihov porast od +0,05 do +0,11 čisti učinak boljeg dohvata. Pitanja na koja se može odgovoriti ima 37–41 od 50, prema 29–33 u ranijem runu. Answerable ovisi o modelu jer model sam klasificira upit i izvlači entitete.

---

## Stari sudac prema novim metrikama

| Spearman ρ | Vrijednost |
|---|---|
| Stari sudac prema gold Scoreu, razina odgovora (n = 450) | 0,45 |
| Stari sudac prema gold Scoreu, razina modela (9 modela) | 0,87 |
| Isto, samo 5 lokalnih modela | 0,25 |
| Faithful prema gold Scoreu, razina modela | 0,67 |
| Faithful prema gold Scoreu, razina odgovora (n = 447) | 0,15 |

- **Stari sudac dobro odvaja cloud od lokalnih modela, ali precjenjuje tu razliku.** gpt-oss-20b i qwen2.5:7b razlikuju se za 0,07 gold Scorea, a kod suca za 1,1 boda od 5. Unutar lokalnih modela sudac ih gotovo ne razlikuje: raspon mu je 3,27–3,49, a phi3:mini, koji je po Scoreu pretposljednji, kod njega je prvi. Vjerojatno nagrađuje stil i duljinu odgovora više od toga što odgovor stvarno citira.
- **Na razini pojedinog odgovora slaganje je umjereno (ρ = 0,45).** Na runovima od 21. 6., sa starim dohvatom, bilo je praktički nula (ρ = 0,04, n = 747). Sudac je dakle nepouzdan upravo kad se ocjenjuje pojedini odgovor.
- **Gold Score i Faithful mjere različite stvari.** Gold Score mjeri koje radove odgovor citira, a Faithful mjeri drži li se tekst odgovora konteksta. Na razini odgovora gotovo ne koreliraju (0,15), a na razini modela umjereno (0,67). Primjer je gpt-oss-20b: najbolje bira radove, ali četvrtina njegovih tvrdnji nije u kontekstu.

> Napomena: raniji preliminarni nalaz (ρ = 0,31; groundedness 3,27 za lažna odbijanja odgovora) izračunat je prije proširenja gold seta i bodovanja presuda. Na sadašnjim podacima se ne ponavlja: lažna odbijanja dobivaju prosječni groundedness 1,6–1,9. Zato ga ne treba citirati.

---

## Analiza i zaključci

### 1. Dohvat je važniji od izbora modela

Poboljšanje dohvata podiglo je Score svih modela osim allama, i to do +0,19 kod llama3. Razlika između najboljeg lokalnog i najboljeg cloud modela nakon toga je 0,07. Lokalni qwen2.5:7b na test skupu (0,779) je iznad gpt-oss-120b (0,745) i gotovo izjednačen s qwen3.8-27b (0,782).

### 2. Veći model nije nužno bolji

gpt-oss-20b je ispred gpt-oss-120b po Scoreu na svih 50 pitanja (0,830 prema 0,798) i na test skupu (0,833 prema 0,745). Po faithfulnessu je redoslijed obrnut (0,694 prema 0,722).

### 3. Preciznost citiranja prema iskorištenju konteksta

qwen3.8-27b citira 98 % relevantnih radova iz konteksta (CiteUtil 0,983), ali uz njih i više nerelevantnih (CitePrec 0,724). Ujedno se najbolje drži konteksta (Faithful 0,814). gpt-oss-20b bira preciznije (CitePrec 0,832), ali dodaje više informacija izvan konteksta.

### 4. Usporedbe metoda ostaju najteži tip upita

`method_comparison` (0,599) i `claim_verification` (0,645) najslabiji su tipovi. Graf rijetko sadrži obje metode iz pitanja na istom zadatku. Kod claim pitanja nije problem dohvat nego zaključak: modeli presudu SUPPORTED ili REFUTED donose iz općeg znanja, a ne iz grafa (npr. Q43: YOLO prema Faster R-CNN).

### 5. allam-2-7b nije upotrebljiv za ovaj zadatak

Odbija odgovoriti na 78 % pitanja na koja se može odgovoriti, a u 32 % odgovora navodi izmišljene naslove. Mali kontekstni prozor dodatno mu ruši pojedina pitanja: Groq vraća `context_length_exceeded`, a na Q16 i Q19 odgovor je prazan. Te greške završavaju u tekstu odgovora i boduju se kao odbijanje odgovora.

### 6. Modeli rijetko proturječe kontekstu, ali često dodaju

Udio C u svim modelima je 1–3 %, ali je podcijenjen jer sudac pogrešna pripisivanja označava kao U. Glavni problem vjernosti je dodavanje: 18–33 % tvrdnji nije u kontekstu. Riječ je o pozadinskom znanju, objašnjenjima metoda i podacima iz jednog rada pripisanima drugome.

---

## Ograničenja

- **Gold set i presude izradio je Claude, a autor ih je pregledao.** Za 35 radova koji nisu ušli u kontekst modela, nego samo u listu izvora (sva su to claim pitanja), ocjena je dana samo prema naslovu. Ta pitanja se boduju presudom, pa ovaj utjecaj na Score je malen.
- **Verdict počiva na 7 pitanja**, a na test skupu na 3. Te brojke su orijentacijske.
- **Fabr je heuristika** (sličnost naslova > 0,8), pa može dati lažno pozitivne rezultate.
- **Faithfulness sudac ima κ = 0,51.** Pouzdan je za prosjeke po modelu, ali ne za pojedinačne tvrdnje. Validaciju je napravio jači model uz ljudski pregled samo za neslaganja.
- **Jedan run po modelu.** Generacija nije potpuno deterministička, a dohvat ovisi o tome kako model klasificira upit. Raspršenje između runova nije mjereno.
- **Ovo je Groq free tier.** Dostupnost modela se mijenja (`llama-3.3-70b` i `llama-4-scout` su povučeni), pa usporedba s `BENCHMARK_REPORT.md` nije potpuna.

---

## Reprodukcija

```bash
python -m eval.answer_eval --results eval/results/benchmark_20261019_105454.json eval/results/benchmark_20261019_121003.json
python -m eval.answer_eval --results ... --split test --by-type
python -m eval.faithfulness --results eval/results/benchmark_20261019_121003.json   # nastavlja gdje je stao
python -m eval.faithfulness --agreement eval/results/faithfulness_benchmark_20261019_121003_sample.json
```

## Datoteke rezultata

| Datoteka | Sadržaj |
|---|---|
| `eval/results/benchmark_20261019_105454.json` | lokalni modeli, odgovori + kontekst + ocjene starog suca |
| `eval/results/benchmark_20261019_121003.json` | Groq modeli, isto |
| `eval/results/faithfulness_benchmark_20261019_*.json` | tvrdnje i S/U/C oznake po odgovoru |
| `eval/results/faithfulness_benchmark_20261019_*_sample.json` | uzorak za validaciju (`judge_label`, `claude_label`, `human_label`) |
| `eval/results/faithfulness_20261019_review.md` | 21 neslaganje s obrazloženjem |
| `data/gold/retrieval_gold.json` | graded gold set (uključuje ocjene `benchmark_20261019`) |
| `data/gold/claim_verdicts.json` | gold presude za claim pitanja |
