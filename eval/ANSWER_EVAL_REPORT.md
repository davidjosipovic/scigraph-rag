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

Na istim odgovorima stari sudac daje cloud modelima 4,3–4,6, a lokalnima 3,3–3,5 (od 5). To je razlika od oko 22 % skale. Gold Score između najboljeg cloud i najboljeg lokalnog modela razlikuje se za 7 postotnih bodova. Unutar lokalnih modela sudac gotovo ne razlikuje kvalitetu: Spearman ρ s gold Scoreom je 0,20.

U usporedbi s vektorskim RAG-om (`mxbai-embed-large`, isti pipeline, mijenja se samo izbor radova) graf ima jednak ukupni dohvat (nDCG@8 0,716 prema 0,725). Graf je bolji kod dataseta, usporedbi metoda i provjere tvrdnji, a vektor kod tema i traženja rada. Dohvaćeni skupovi gotovo se ne preklapaju, pa hibrid doseže 0,79; u stvarnom pipelineu hibrid značajno poboljšava dohvat u odnosu na graf (0,746 prema 0,682), ali ne i odgovore. U odgovorima ukupne razlike nema (strogi E2E: razlika oko 0 ± 0,1 za dva modela), a raspršenje između runova jednako je velikoj razlici između sustava. Graf je dosljedno bolji kod usporedbi metoda i malo vjerniji kontekstu.

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

- Sudac griješi u oba smjera, 7 puta prestrogo i 9 puta preblago (7 U→S i 2 C→S). Udio S u uzorku je zato gotovo isti (sudac 73 %, referenca 71 %), pa su prosjeci po modelu tek malo pristrani. Pojedinačne oznake su umjereno pouzdane.
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
| Isto, samo 5 lokalnih modela | 0,20 |
| Faithful prema gold Scoreu, razina modela | 0,67 |
| Faithful prema gold Scoreu, razina odgovora (n = 447) | 0,15 |

- **Stari sudac dobro odvaja cloud od lokalnih modela, ali precjenjuje tu razliku.** gpt-oss-20b i qwen2.5:7b razlikuju se za 0,07 gold Scorea, a kod suca za 1,1 boda od 5. Unutar lokalnih modela sudac ih gotovo ne razlikuje: raspon mu je 3,27–3,49, a phi3:mini, koji je po Scoreu pretposljednji, kod njega je prvi. Ocjena suca raste s duljinom odgovora (Spearman ρ = 0,26 bez allama, 0,16 unutar istog modela), a gold Score ne (ρ = −0,01); phi3:mini piše najduže lokalne odgovore.
- **Na razini pojedinog odgovora slaganje je umjereno (ρ = 0,45).** Na runovima od 21. 6., sa starim dohvatom, bilo je praktički nula (ρ = 0,04, n = 747). Sudac je dakle nepouzdan upravo kad se ocjenjuje pojedini odgovor.
- **Gold Score i Faithful mjere različite stvari.** Gold Score mjeri koje radove odgovor citira, a Faithful mjeri drži li se tekst odgovora konteksta. Na razini odgovora gotovo ne koreliraju (0,15), a na razini modela umjereno (0,67). Primjer je gpt-oss-20b: najbolje bira radove, ali četvrtina njegovih tvrdnji nije u kontekstu.

**Sudac i usporedba dohvata.** Isti sudac (qwen3:8b, vidi samo naslove izvora) naknadno je ocijenio i vektorske, hibridne i drugi run grafa (`eval/rejudge.py`). Za qwen2.5:7b daje vektorskim odgovorima 0,57 [0,35; 0,79] boda više nego odgovorima s grafom (graf 3,46 / 3,45, vektor 4,00 / 4,05, hibrid 3,98 / 3,99), i to čak i kod `method_comparison` (vektor 3,7–3,9, graf 3,0–3,2), gdje gold metrike daju prednost grafu (E2E 0,60–0,64 prema 0,37–0,51). Za gpt-oss-20b razlika je −0,12 [−0,28; 0,02]. Evaluacija samo ovim sucem zaključila bi da je vektorski dohvat jasno bolji.

> Napomena: raniji preliminarni nalaz (ρ = 0,31; groundedness 3,27 za lažna odbijanja odgovora) izračunat je prije proširenja gold seta i bodovanja presuda. Na sadašnjim podacima se ne ponavlja: lažna odbijanja dobivaju prosječni groundedness 1,6–1,9. Zato ga ne treba citirati.

---

## GraphRAG prema vektorskom RAG-u

Sustav ostaje čisti GraphRAG, bez embeddinga. Vektorski RAG ovdje služi samo kao usporedna točka za evaluaciju (`eval/vector_baseline.py`), da se vidi što graf donosi u odnosu na standardni pristup.

### Postav

`VectorPipeline` nasljeđuje `RAGPipeline` i mijenja samo izbor 8 radova. Sve ostalo je isto: klasifikacija upita, ORKG zapisi radova koji ulaze u kontekst, odabir redaka, prompt i model. Razlika u odgovorima zato dolazi isključivo od dohvata.

- **Indeks:** svih 68 806 radova iz javnog ORKG RDF dumpa (`data/orkg_dump_20261018.nt`). Svaki rad ima jednu karticu teksta od najviše 1 500 znakova: naslov, istraživačko polje i oznake vrijednosti iz doprinosa, s istraživačkim problemima na početku.
- **Embedding:** `mxbai-embed-large` (1 024 dimenzije) preko Ollame, s instrukcijskim prefiksom na strani upita. Rangiranje je kosinusnom sličnošću.
- **Gold set:** vektor je dohvatio 298 radova koji nisu bili ocijenjeni. Claude ih je ocijenio prema kriterijima pitanja (`found_by: vector_baseline`). Za drugi model, `nomic-embed-text`, ocijenjeno je još 310 radova (`found_by: vector_nomic`). Nakon toga nema neocijenjenih radova ni u jednom dohvatu.
- **Provjera gold seta:** autor je naslijepo ocijenio 120 parova pitanje–rad, po 40 za svaku ocjenu (`gold_kappa_sample_labelled.json`). Slaganje s gold setom je κ = 0,78, a kvadratno ponderirani κ 0,85. Na radovima koje je donio vektor κ je 0,91, pa nove ocjene nisu pristrane u korist ijednog sustava.

### Dohvat (nDCG@8)

| Sustav | Svih 50 | Test (25) |
|---|---|---|
| Graf, prvotni dohvat (A) | 0,496 | 0,527 |
| Graf, konačni (oracle tip i entiteti) | 0,716 | **0,754** |
| Graf, konačni (end-to-end, qwen2.5:7b) | – | 0,695 |
| Vektor, `mxbai-embed-large` | **0,725** | 0,712 |
| Vektor, `nomic-embed-text` | 0,413 | 0,435 |
| Hibrid RRF (graf oracle + mxbai), procjena | – | **0,792** |

| Tip upita (svih 50) | Graf (oracle) | Vektor (mxbai) |
|---|---|---|
| `dataset_search` | **0,817** | 0,688 |
| `claim_verification` | **0,657** | 0,563 |
| `method_comparison` | **0,660** | 0,535 |
| `method_usage` | 0,710 | **0,817** |
| `paper_lookup` | 0,583 | **0,724** |
| `topic_search` | 0,850 | **0,979** |

- **Ukupno su graf i jak vektorski dohvat izjednačeni.** Upareni bootstrap interval razlike uključuje 0.
- **Prednosti su komplementarne.** Graf je bolji gdje pitanje ima strukturu, poput dataseta, para metoda ili tvrdnje o rezultatu. Tada SPARQL strategije pogađaju točne veze u grafu. Vektor je bolji na temama i traženju rada po naslovu, gdje je dovoljna tekstualna sličnost.
- **Sustavi dohvaćaju gotovo disjunktne skupove:** u prosjeku se u 8 dohvaćenih radova poklapa 0,42 rada. Zato hibrid s RRF-om (0,792) nadmašuje oba. To je procjena nad listama (graf oracle); hibrid pokrenut u pipelineu opisan je u poglavlju "Hibrid u pipelineu".
- **Vektorski rezultat jako ovisi o embedding modelu.** S `nomic-embed-text` nDCG pada za 0,3. Prednost vektora na temama ostaje (0,866), ali na usporedbama metoda pada na 0,166. Kao usporedna točka uzet je jači model, mxbai.
- **Gubitak end-to-end** (0,754 → 0,695) dolazi od toga što LLM pogrešno klasificira upit ili izvuče krive entitete. Vektorski dohvat te korake ne koristi, pa taj gubitak nema.

### Odgovori

Za usporedbu dohvata blagi gold Score nije prikladan. Na claim pitanjima prihvaća INSUFFICIENT kad ključni radovi nisu dohvaćeni, a kontekst bez relevantnih radova nagrađuje odbijanje odgovora. Lošiji dohvat zato može dobiti *viši* Score. Primjer je qwen2.5:7b: vektor 0,809 prema grafu 0,758, iako vektor rjeđe dohvaća ključne radove.

Zato se koristi **strogi end-to-end Score (E2E)**:
- odbijanje odgovora ne donosi bodove
- claim pitanja boduju se samo točnom presudom, INSUFFICIENT se ne prihvaća
- ostala pitanja boduju se s F1(CitePrec, CiteUtil)

qwen2.5:7b pokrenut je dvaput, da se izmjeri raspršenje između runova.

| Model, run | E2E graf | E2E vektor | Faithful graf | Faithful vektor |
|---|---|---|---|---|
| qwen2.5:7b, run 1 | 0,748 | 0,731 | 0,771 | 0,739 |
| qwen2.5:7b, run 2 | 0,696 | 0,733 | 0,758 | 0,733 |
| gpt-oss-20b | 0,745 | 0,745 | 0,688 | 0,640 |
| **Razlika graf − vektor, upareni bootstrap 95 % CI** | | | | |
| qwen2.5:7b, oba runa zajedno | −0,01 [−0,10; 0,08] | | +0,03 [−0,03; 0,09] | |
| gpt-oss-20b | 0,00 [−0,11; 0,11] | | +0,05 [−0,04; 0,13] | |

Faithful za gpt-oss-20b računat je na 49 odgovora (Q01 na vektorskoj strani sudac nije ocijenio), a za qwen2.5:7b na 46 zajedničkih odgovora.

| Tip upita, E2E | qwen2.5 graf (run 1 / 2) | qwen2.5 vektor (run 1 / 2) | gpt-oss-20b graf | gpt-oss-20b vektor |
|---|---|---|---|---|
| `method_comparison` | **0,64 / 0,60** | 0,37 / 0,51 | **0,68** | 0,42 |
| `claim_verification` | 0,62 / 0,50 | 0,62 / 0,62 | 0,62 | 0,62 |
| `paper_lookup` | 0,74 / 0,65 | **0,82 / 0,84** | 0,61 | **0,80** |
| `topic_search` | 0,87 / 0,87 | 0,93 / 0,99 | 0,97 | 0,99 |

| Citirani radovi ocjene 2 po pitanju | qwen2.5 graf (run 1 / 2) | qwen2.5 vektor (run 1 / 2) | gpt-oss-20b graf | gpt-oss-20b vektor |
|---|---|---|---|---|
| `claim_verification` | **2,25 / 2,38** | 0,38 / 1,25 | **2,12** | 1,38 |
| `method_comparison` | **2,00 / 1,88** | 0,75 / 1,00 | **2,25** | 1,00 |

- **Ukupna kvaliteta odgovora ne razlikuje se.** Kod oba modela razlika u E2E je oko 0, uz interval od ±0,1.
- **Raspršenje između runova jednako je velikoj razlici između sustava.** Kod qwen2.5 dohvaćeni izvori su u oba runa gotovo isti (49 od 50 pitanja za graf, 50 od 50 za vektor). Ipak se E2E grafa pomaknuo s 0,748 na 0,696, a predznak razlike graf − vektor se promijenio. Pojedini odgovor se između runova u prosjeku pomakne za 0,08–0,09. Razlike od nekoliko postotnih bodova iz jednog runa zato ne treba tumačiti.
- **Prednost grafa na strukturiranim pitanjima prenosi se u odgovore.** Kod usporedbe metoda graf vodi po E2E u sva tri mjerenja. Kod usporedbe metoda i provjere tvrdnji odgovori s grafom citiraju 1,5 do 6 puta više ključnih radova (ocjena 2). Točnost presude na claim pitanjima ipak nije veća, jer modeli presudu donose iz općeg znanja (vidi zaključak 4).
- **Kod traženja konkretnog rada vektor je bolji** i u dohvatu i u odgovorima.
- **Faithfulness je viši s grafom u sva tri mjerenja**, za +0,03 do +0,05. Pojedinačno nijedna razlika nije značajna. Moguće objašnjenje je da graf donosi radove čiji zapisi izravno odgovaraju na pitanje, pa model manje dodaje iz vlastitog znanja. Ovo nije provjereno.


### Hibrid u pipelineu

`HybridPipeline` (`eval/vector_baseline.py`, `answer --hybrid`) spaja top 8 grafa (end-to-end, entitete izvlači qwen2.5:7b) i top 8 vektora (mxbai) RRF-om (k = 60) i uzima 8 radova. Radovi iz grafa zadržavaju retke koje im je dao graf, a radovi koje je našao samo vektor dobivaju retke iz svog ORKG zapisa, kao u vektorskom pipelineu. qwen2.5:7b pokrenut je dvaput; u oba runa dohvaćeni su isti radovi i svi su ocijenjeni.

| qwen2.5:7b | nDCG@8 izvora (svih 50 / test) | E2E (run 1 / 2) | Gold Score | Faithful |
|---|---|---|---|---|
| Graf | 0,682 / 0,695 | 0,748 / 0,696 | 0,758 / 0,726 | 0,771 / 0,758 |
| Vektor | 0,711 / 0,701 | 0,731 / 0,733 | 0,809 / 0,811 | 0,739 / 0,733 |
| **Hibrid** | **0,746 / 0,743** | 0,736 / 0,752 | 0,766 / 0,782 | 0,714 / 0,725 |

nDCG je ovdje računat iz 8 izvora u kontekstu stvarnog odgovora, pa se za vektor malo razlikuje od `retrieval_vector_*` (0,725 / 0,712).

| Razlika, upareni bootstrap 95 % | |
|---|---|
| Hibrid − graf, nDCG | **+0,064 [+0,016; +0,113]** |
| Hibrid − vektor, nDCG | +0,034 [−0,027; +0,098] |
| Hibrid − graf, E2E (oba runa) | +0,022 [−0,058; +0,102] |
| Hibrid − vektor, E2E (oba runa) | +0,012 [−0,049; +0,079] |
| Hibrid − graf, faithful (oba runa, n = 46) | −0,039 [−0,084; +0,004] |
| Hibrid − vektor, faithful (oba runa, n = 46) | −0,014 [−0,069; +0,044] |

- **Dohvat:** hibrid je jedini sustav koji je značajno bolji od drugoga (od grafa end-to-end). Ispod je procjene od 0,792 jer graf ovdje radi s entitetima koje izvlači LLM, a ne s oracle entitetima.
- **Odgovori:** E2E je najviši u prosjeku dva runa (0,744), ali unutar šuma. Na `claim_verification` hibrid ima 0,75 u oba runa (graf 0,50–0,62, vektor 0,62), a kod `method_comparison` pada na 0,47–0,51 (graf 0,60–0,64). To nije zbog dohvata: hibridni kontekst ima jednako ključnih radova kao graf (3,0 prema 2,9 po pitanju; vektorski radovi nadoknade 7 od 23 izgubljena rada iz grafa), ali odgovori ih citiraju manje (1,5–1,6 prema 1,9–2,0). Zašto, nije istraženo.
- **Faithfulness** je na razini vektora i nešto ispod grafa; razlika prema grafu nije značajna, ali je graf ostao najvjerniji u svim runovima.

**gpt-oss-20b** (jedan run, `benchmark_hybrid_20261020_121456.json`; Q35–Q50 ponovljeni 21. 10. nakon Groq limita):

| gpt-oss-20b | E2E | Gold Score | Faithful (n = 49) | Cl. (radovi ocjene 2) | E2E `method_comparison` | E2E `claim_verification` |
|---|---|---|---|---|---|---|
| Graf | 0,745 | 0,830 | 0,688 | 2,12 | **0,68** | 0,62 |
| Vektor | 0,745 | 0,841 | 0,640 | 1,38 | 0,42 | 0,62 |
| Hibrid | 0,739 | 0,802 | 0,672 | 1,75 | 0,54 | 0,50 |

| Razlika, upareni bootstrap 95 % | |
|---|---|
| Hibrid − graf, E2E | −0,007 [−0,095; +0,079] |
| Hibrid − vektor, E2E | −0,006 [−0,092; +0,083] |
| Hibrid − graf, faithful (n = 49) | −0,016 [−0,076; +0,043] |
| Hibrid − vektor, faithful (n = 49) | +0,032 [−0,037; +0,103] |

- Ni kod gpt-oss-20b hibrid ne poboljšava odgovore; sve su razlike unutar šuma.
- Prednost hibrida na `claim_verification` (0,75 kod qwen2.5) ovdje se ne ponavlja (0,50), pa je vjerojatno slučajna (8 pitanja).
- Kod `method_comparison` hibrid je opet između grafa i vektora, a faithfulness također.

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
- **Jedan run po modelu u općem poretku.** Raspršenje je izmjereno samo za qwen2.5:7b, s dva runa (vidi "GraphRAG prema vektorskom RAG-u"). Pojedini odgovor se između runova pomakne za 0,08–0,09 E2E, a prosjek za do 0,05. Razlike između susjednih modela u općem poretku (npr. 0,798 i 0,790) zato su unutar šuma.
- **Usporedba s vektorskim RAG-om ima jedan graf, jedan korpus i dva embedding modela.** Hibrid je u pipelineu pokrenut samo s qwen2.5:7b.
- **Ovo je Groq free tier.** Dostupnost modela se mijenja (`llama-3.3-70b` i `llama-4-scout` su povučeni), pa usporedba s `BENCHMARK_REPORT.md` nije potpuna.

---

## Reprodukcija

```bash
python -m eval.answer_eval --results eval/results/benchmark_20261019_105454.json eval/results/benchmark_20261019_121003.json
python -m eval.answer_eval --results ... --split test --by-type
python -m eval.faithfulness --results eval/results/benchmark_20261019_121003.json   # nastavlja gdje je stao
python -m eval.faithfulness --agreement eval/results/faithfulness_benchmark_20261019_121003_sample.json

# vektorski RAG
python -m eval.vector_baseline build --dump data/orkg_dump_20261018.nt && python -m eval.vector_baseline embed
python -m eval.vector_baseline retrieve [--split test]
python -m eval.vector_baseline answer --models ollama:qwen2.5:7b [--questions Q38 Q39 ...]
VECTOR_EMBED_MODEL=nomic-embed-text python -m eval.vector_baseline embed   # drugi embedding model
```

## Datoteke rezultata

| Datoteka | Sadržaj |
|---|---|
| `eval/results/benchmark_20261019_105454.json` | lokalni modeli, odgovori + kontekst + ocjene starog suca |
| `eval/results/benchmark_20261019_121003.json` | Groq modeli, isto |
| `eval/results/faithfulness_benchmark_20261019_*.json` | tvrdnje i S/U/C oznake po odgovoru |
| `eval/results/faithfulness_benchmark_20261019_*_sample.json` | uzorak za validaciju (`judge_label`, `claude_label`, `human_label`) |
| `eval/results/faithfulness_20261019_review.md` | 21 neslaganje s obrazloženjem |
| `eval/results/benchmark_20261020_000036.json` | qwen2.5:7b, graf, run 2 |
| `eval/results/benchmark_vector_20261019_220522.json` | vektor (mxbai): qwen2.5:7b run 1 i gpt-oss-20b |
| `eval/results/benchmark_vector_20261020_002631.json` | vektor (mxbai): qwen2.5:7b run 2 |
| `eval/results/benchmark_hybrid_20261020_*.json` | hibrid (RRF): qwen2.5:7b run 1 i 2 |
| `eval/results/faithfulness_benchmark_*2026102*.json`, `faithfulness_benchmark_vector_*.json` | faithfulness za gornje runove |
| `eval/results/retrieval_vector_*.json`, `retrieval_vector_nomic_*.json`, `retrieval_oracle_20261019_*.json` | nDCG po pitanju za vektor i graf |
| `eval/results/gold_kappa_sample_labelled.json` | 120 parova koje je autor naslijepo ocijenio |
| `data/gold/retrieval_gold.json` | graded gold set (uključuje ocjene `benchmark_20261019`, `vector_baseline`, `vector_nomic`) |
| `data/gold/claim_verdicts.json` | gold presude za claim pitanja |
