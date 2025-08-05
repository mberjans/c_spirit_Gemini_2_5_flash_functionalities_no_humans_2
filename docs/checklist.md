.

## **Software Development Checklist: Ontology Development and Information Extraction**

This checklist provides granular tasks for each ticket, guiding the development team through the process with a test-driven approach.

### **1\. Core Project Setup & Standards**

Ticket ID: AIM2-ODIE-001  
Description: Project Setup & Version Control Initialization: Initialize Git repository, set up basic project directory structure (src/, data/, tests/, docs/), and create initial README.md.  
Dependencies: None  
Independent: Yes

- [x] **AIM2-ODIE-001-T1:** Initialize Git repository in the project root.  
- [x] **AIM2-ODIE-001-T2:** Create src/ directory for source code.  
- [x] **AIM2-ODIE-001-T3:** Create data/ directory for input/output data.  
- [x] **AIM2-ODIE-001-T4:** Create tests/ directory for unit and integration tests.  
- [x] **AIM2-ODIE-001-T5:** Create docs/ directory for documentation.  
- [x] **AIM2-ODIE-001-T6:** Create initial README.md file with a brief project overview.

Ticket ID: AIM2-ODIE-002  
Description: Dependency Management with Poetry: Configure pyproject.toml for project metadata and initial dependencies (e.g., Owlready2, Biopython, PyMuPDF, text2term, LLM-IE, OntoGPT, FuzzyWuzzy, dedupe, multitax, ncbi-taxonomist, pytest, ruff, black). Set up poetry.lock.  
Dependencies: AIM2-ODIE-001  
Independent: No

- [x] **AIM2-ODIE-002-T1:** **Develop Unit Tests:** Write unit tests (tests/test\_dependencies.py) to verify Poetry installation and basic dependency imports (e.g., poetry run python \-c "import owlready2").  
- [x] **AIM2-ODIE-002-T2:** Install Poetry on the development environment if not already present.  
- [x] **AIM2-ODIE-002-T3:** Initialize Poetry project in the root directory (poetry init).  
- [x] **AIM2-ODIE-002-T4:** Add core runtime dependencies to pyproject.toml (e.g., Owlready2, Biopython, PyMuPDF, text2term, LLM-IE, OntoGPT, FuzzyWuzzy, dedupe, multitax, ncbi-taxonomist).  
- [x] **AIM2-ODIE-002-T5:** Add development dependencies to pyproject.toml (e.g., pytest, ruff, black).  
- [x] **AIM2-ODIE-002-T6:** Run poetry install to generate poetry.lock and install all specified dependencies.  
- [x] **AIM2-ODIE-002-T7:** **Conduct Unit Tests:** Run unit tests developed in T1 to confirm successful dependency setup.

Ticket ID: AIM2-ODIE-003  
Description: Establish Code Quality & Testing Standards: Integrate ruff (or flake8) and black for linting and formatting. Set up pytest for unit/integration testing. Define initial testing framework module for wrapped utilities and fakers.  
Dependencies: AIM2-ODIE-002  
Independent: No

- [x] **AIM2-ODIE-003-T1:** **Develop Unit Tests:** Write a dummy Python file (src/temp\_test\_file.py) with intentional linting/formatting errors and a simple pytest test case. Write a test (tests/test\_code\_quality.py) to run ruff/black checks and pytest on this dummy file.  
- [x] **AIM2-ODIE-003-T2:** Configure ruff (or flake8) in pyproject.toml or a dedicated config file (e.g., .ruff.toml).  
- [x] **AIM2-ODIE-003-T3:** Configure black in pyproject.toml or a dedicated config file (e.g., pyproject.toml).  
- [x] **AIM2-ODIE-003-T4:** Create tests/conftest.py for pytest configuration (if needed for shared fixtures).  
- [x] **AIM2-ODIE-003-T5:** Create src/utils/testing\_framework.py to encapsulate pytest.raises, pytest.mark.parametrize, and freezegun.freeze\_time (if freezegun is added as a dev dependency).  
- [x] **AIM2-ODIE-003-T6:** Implement initial "Fakers" (e.g., fake\_text, fake\_entity) within src/utils/testing\_framework.py for common data types.  
- [x] **AIM2-ODIE-003-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-004  
Description: Document Non-Python Dependencies: Create a docs/INSTALL.md or similar document detailing manual installation steps for non-Python dependencies (e.g., Java for HermiT/Pellet reasoners in Owlready2, ollama for local LLMs).  
Dependencies: AIM2-ODIE-001  
Independent: Yes

- [x] **AIM2-ODIE-004-T1:** Create docs/INSTALL.md file.  
- [x] **AIM2-ODIE-004-T2:** Document Java Development Kit (JDK) installation instructions, specifying minimum version required for Owlready2 reasoners (HermiT/Pellet).  
- [x] **AIM2-ODIE-004-T3:** Document ollama installation instructions for running local LLM models.  
- [x] **AIM2-ODIE-004-T4:** Add a section for any other known non-Python dependencies that might arise (e.g., Graphviz if graph visualization were to be added later).  
- [x] **AIM2-ODIE-004-T5:** Include a disclaimer about potential system-specific variations and troubleshooting tips for non-Python dependencies.

### **2\. Ontology Development and Management**

Ticket ID: AIM2-ODIE-005  
Description: Ontology Loading Module: Develop a Python module (src/ontology/loader.py) to load OWL 2.0 ontologies using Owlready2 (get\_ontology().load()) from URLs or local files. Implement basic error handling for loading failures.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [x] **AIM2-ODIE-005-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_loader.py) for src/ontology/loader.py to cover:  
  * Successful loading of a valid local OWL file.  
  * Successful loading of a valid OWL file from a URL (mock external request using pytest-mock or similar).  
  * Error handling for non-existent local files (e.g., FileNotFoundError).  
  * Error handling for invalid URLs or network issues (e.g., requests.exceptions.ConnectionError).  
  * Error handling for invalid OWL file formats (e.g., Owlready2 parsing errors).  
- [x] **AIM2-ODIE-005-T2:** Create src/ontology/loader.py.  
- [x] **AIM2-ODIE-005-T3:** Implement load\_ontology\_from\_file(file\_path: str) function using Owlready2.get\_ontology(f"file://{file\_path}").load().  
- [x] **AIM2-ODIE-005-T4:** Implement load\_ontology\_from\_url(url: str) function using Owlready2.get\_ontology(url).load().  
- [x] **AIM2-ODIE-005-T5:** Add try-except blocks to catch relevant exceptions during ontology loading and re-raise custom, more informative exceptions.  
- [x] **AIM2-ODIE-005-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-006  
Description: Ontology Trimming & Filtering Core Logic: Implement core logic (src/ontology/trimmer.py) for programmatic trimming and filtering of ontology terms based on criteria (e.g., keyword matching, hierarchical relationships, specific properties) using Owlready2's search() and iteration methods.  
Dependencies: AIM2-ODIE-005  
Independent: No

- [x] **AIM2-ODIE-006-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_trimmer.py) for src/ontology/trimmer.py to cover:  
  * Filtering classes by keyword in their name or label using ontology.search().  
  * Filtering individuals based on specific property values.  
  * Filtering subclasses of a given base class using is\_a or subclass\_of in search().  
  * Filtering based on a combination of criteria (e.g., class name AND property value).  
  * Ensuring the original ontology object is not modified if a "copy" operation is implied by the filtering.  
- [x] **AIM2-ODIE-006-T2:** Create src/ontology/trimmer.py.  
- [x] **AIM2-ODIE-006-T3:** Implement filter\_classes\_by\_keyword(ontology, keyword: str) function.  
- [x] **AIM2-ODIE-006-T4:** Implement filter\_individuals\_by\_property(ontology, property\_name: str, value: Any) function.  
- [x] **AIM2-ODIE-006-T5:** Implement get\_subclasses(ontology, base\_class\_iri: str) function.  
- [x] **AIM2-ODIE-006-T6:** Implement a general apply\_filters(ontology, filters: dict) function that combines multiple filtering criteria.  
- [x] **AIM2-ODIE-006-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-007  
Description: Ontology Entity Deletion Functionality: Implement functions (src/ontology/editor.py) to programmatically delete irrelevant classes, individuals, or properties using Owlready2's destroy\_entity() function.  
Dependencies: AIM2-ODIE-005  
Independent: No

- [x] **AIM2-ODIE-007-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_editor.py) for src/ontology/editor.py to cover:  
  * Deletion of a specific class and verification of its absence using ontology.search\_one().  
  * Deletion of a specific individual and verification of its absence.  
  * Deletion of a property and verification of its absence.  
  * Verification that associated relations/constructs are also removed upon entity deletion (e.g., if a class is deleted, its instances are also gone).  
  * Error handling for attempting to delete non-existent entities.  
- [x] **AIM2-ODIE-007-T2:** Create src/ontology/editor.py.  
- [x] **AIM2-ODIE-007-T3:** Implement delete\_class(ontology, class\_iri: str) function using destroy\_entity().  
- [x] **AIM2-ODIE-007-T4:** Implement delete\_individual(ontology, individual\_iri: str) function using destroy\_entity().  
- [x] **AIM2-ODIE-007-T5:** Implement delete\_property(ontology, property\_iri: str) function using destroy\_entity().  
- [x] **AIM2-ODIE-007-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-008  
Description: Ontology Export Functionality: Develop a function (src/ontology/exporter.py) to export the refined ontology to OWL/RDF/XML format.  
Dependencies: AIM2-ODIE-005  
Independent: No

- [x] **AIM2-ODIE-008-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_exporter.py) for src/ontology/exporter.py to cover:  
  * Successful export of a loaded ontology to a specified temporary file path.  
  * Verification that the exported file is not empty and contains expected OWL/RDF/XML tags.  
  * Attempting to load the exported file back into Owlready2 to confirm its validity and integrity.  
  * Error handling for invalid file paths or write permissions.  
- [x] **AIM2-ODIE-008-T2:** Create src/ontology/exporter.py.  
- [x] **AIM2-ODIE-008-T3:** Implement export\_ontology(ontology, file\_path: str, format: str \= 'rdfxml') function using ontology.save(file=file\_path, format=format).  
- [x] **AIM2-ODIE-008-T4:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-009  
Description: Refined Ontology Scheme Definition (Structural): Programmatically define and integrate terms for "Structural Annotation" (Chemont classification, NP Classifier, Plant Metabolic Network) into the core ontology using Owlready2.  
Dependencies: AIM2-ODIE-005, AIM2-ODIE-007  
Independent: No

- [x] **AIM2-ODIE-009-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_scheme\_structural.py) for src/ontology/scheme\_structural.py to cover:  
  * Creation of new Owlready2 classes representing Chemont, NP Classifier, and PMN categories within the target ontology.  
  * Verification that these classes are correctly added and accessible in the ontology.  
  * Verification of basic hierarchical relationships (e.g., is\_a) if defined within this scheme (e.g., NPClass is a subclass of ChemicalClass).  
- [x] **AIM2-ODIE-009-T2:** Create src/ontology/scheme\_structural.py.  
- [x] **AIM2-ODIE-009-T3:** Define Python classes for core structural annotation concepts (e.g., ChemontClass, NPClass, PMNCompound) inheriting from owlready2.Thing and associating them with the main ontology namespace.  
- [x] **AIM2-ODIE-009-T4:** Programmatically add initial key terms/instances from Chemont, NP Classifier, and PMN (as representative examples) to the ontology.  
- [x] **AIM2-ODIE-009-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-010  
Description: Refined Ontology Scheme Definition (Source): Programmatically define and integrate terms for "Source Annotation" (Plant Ontology, NCBI Taxonomy, PECO) into the core ontology using Owlready2.  
Dependencies: AIM2-ODIE-005, AIM2-ODIE-007  
Independent: No

- [x] **AIM2-ODIE-010-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_scheme\_source.py) for src/ontology/scheme\_source.py to cover:  
  * Creation of new Owlready2 classes representing Plant Ontology, NCBI Taxonomy, and PECO categories.  
  * Verification that these classes are correctly added and accessible in the ontology.  
  * Verification of basic hierarchical relationships (e.g., is\_a) if defined within this scheme (e.g., Root is a subclass of PlantAnatomy).  
- [x] **AIM2-ODIE-010-T2:** Create src/ontology/scheme\_source.py.  
- [x] **AIM2-ODIE-010-T3:** Define Python classes for core source annotation concepts (e.g., PlantAnatomy, Species, ExperimentalCondition) inheriting from owlready2.Thing and associating them with the main ontology namespace.  
- [x] **AIM2-ODIE-010-T4:** Programmatically add initial key terms/instances from Plant Ontology, NCBI Taxonomy, and PECO (as representative examples) to the ontology.  
- [x] **AIM2-ODIE-010-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-011  
Description: Refined Ontology Scheme Definition (Functional): Programmatically define and integrate terms for "Functional Annotation" (Gene Ontology, Trait Ontology, ChemFont) into the core ontology using Owlready2.  
Dependencies: AIM2-ODIE-005, AIM2-ODIE-007  
Independent: No

- [ ] **AIM2-ODIE-011-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_scheme\_functional.py) for src/ontology/scheme\_functional.py to cover:  
  * Creation of new Owlready2 classes representing GO, Trait Ontology, and ChemFont categories.  
  * Verification that these classes are correctly added and accessible in the ontology.  
  * Verification of basic hierarchical relationships (e.g., is\_a) if defined within this scheme (e.g., DroughtTolerance is a subclass of PlantTrait).  
- [ ] **AIM2-ODIE-011-T2:** Create src/ontology/scheme\_functional.py.  
- [ ] **AIM2-ODIE-011-T3:** Define Python classes for core functional annotation concepts (e.g., MolecularTrait, PlantTrait, HumanTrait) inheriting from owlready2.Thing and associating them with the main ontology namespace.  
- [ ] **AIM2-ODIE-011-T4:** Programmatically add initial key terms/instances from GO, Trait Ontology, and ChemFont (as representative examples) to the ontology.  
- [ ] **AIM2-ODIE-011-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-012  
Description: Hierarchical Relationship Definition & Management: Implement logic to define and manage hierarchical relationships ("is\_a", "made\_via", "accumulates\_in", "affects") using Owlready2's ObjectProperty and DataProperty classes, including domain, range, and inverse\_property.  
Dependencies: AIM2-ODIE-005, AIM2-ODIE-009, AIM2-ODIE-010, AIM2-ODIE-011  
Independent: No

- [ ] **AIM2-ODIE-012-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_relationships.py) for src/ontology/relationships.py to cover:  
  * Definition of ObjectProperty classes (e.g., made\_via, accumulates\_in, affects) with correct domain and range specified.  
  * Definition of DataProperty classes if applicable (e.g., has\_molecular\_weight).  
  * Definition of inverse\_property for relevant relationships (e.g., is\_accumulated\_in as inverse of accumulates\_in) and verification of Owlready2's automatic handling.  
  * Creation of example instances with these relationships and verification of property values.  
- [ ] **AIM2-ODIE-012-T2:** Create src/ontology/relationships.py.  
- [ ] **AIM2-ODIE-012-T3:** Define ObjectProperty classes for "made\_via", "accumulates\_in", "affects" within the ontology, linking them to the relevant classes defined in AIM2-ODIE-009, \-010, \-011.  
- [ ] **AIM2-ODIE-012-T4:** Set domain and range for each property using Owlready2 syntax.  
- [ ] **AIM2-ODIE-012-T5:** Define inverse properties where logically applicable (e.g., is\_made\_via as inverse of made\_via).  
- [ ] **AIM2-ODIE-012-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-013  
Description: Ontology Reasoning Integration: Integrate Owlready2's reasoning capabilities (HermiT/Pellet) to infer new facts and reclassify instances/classes based on defined relationships.  
Dependencies: AIM2-ODIE-004, AIM2-ODIE-012  
Independent: No

- [ ] **AIM2-ODIE-013-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology/test\_reasoner.py) for src/ontology/reasoner.py to cover:  
  * Loading a small test ontology with implicit facts (e.g., A is\_a B, B is\_a C, then assert A is\_a C after reasoning).  
  * Verification of inferred class memberships for individuals based on property values and restrictions (e.g., if a compound has property has\_structure of type X, and X implies Y, check if compound is classified as Y).  
  * Verification of inferred property values if infer\_property\_values=True is used.  
  * Handling of inconsistent ontologies (expecting OwlReadyInconsistentOntologyError or similar).  
- [ ] **AIM2-ODIE-013-T2:** Create src/ontology/reasoner.py.  
- [ ] **AIM2-ODIE-013-T3:** Implement run\_reasoner(ontology, infer\_property\_values: bool \= False) function using sync\_reasoner().  
- [ ] **AIM2-ODIE-013-T4:** Ensure the Java executable path is correctly configured for Owlready2 to find HermiT/Pellet (referencing AIM2-ODIE-004).  
- [ ] **AIM2-ODIE-013-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-014  
Description: CLI for Ontology Management (Load, Trim, Export): Create a command-line interface for the ontology management module, allowing users to load, trim/filter, and export ontologies via CLI commands.  
Dependencies: AIM2-ODIE-005, AIM2-ODIE-006, AIM2-ODIE-008  
Independent: No

- [ ] **AIM2-ODIE-014-T1:** **Develop Integration Tests:** Write integration tests (tests/cli/test\_ontology\_cli.py) for the CLI:  
  * Test ontology load \<file\_path\> command with a dummy OWL file.  
  * Test ontology trim \<file\_path\> \--keyword \<keyword\> command with filtering criteria on a dummy ontology.  
  * Test ontology export \<input\_file\> \<output\_file\> command to a temporary file and verify output.  
  * Test invalid arguments (e.g., non-existent file, incorrect format) and ensure proper error messages are displayed.  
- [ ] **AIM2-ODIE-014-T2:** Choose a CLI framework (e.g., Typer or Click) and set up the main CLI entry point (src/cli.py).  
- [ ] **AIM2-ODIE-014-T3:** Implement ontology load subcommand, calling functions from src/ontology/loader.py.  
- [ ] **AIM2-ODIE-014-T4:** Implement ontology trim subcommand, calling functions from src/ontology/trimmer.py with appropriate arguments for filtering criteria.  
- [ ] **AIM2-ODIE-014-T5:** Implement ontology export subcommand, calling functions from src/ontology/exporter.py.  
- [ ] **AIM2-ODIE-014-T6:** Add comprehensive help messages for all commands and arguments using the chosen CLI framework's features.  
- [ ] **AIM2-ODIE-014-T7:** **Conduct Integration Tests:** Run integration tests developed in T1.

### **3\. Literature Information Extraction using LLMs**

Ticket ID: AIM2-ODIE-015  
Description: PubMed/PMC Data Acquisition Module: Develop a module (src/data\_acquisition/pubmed.py) using Biopython.Entrez to search and retrieve abstracts/full texts (XML format) from PubMed/PMC based on keywords. Implement rate limiting and error handling.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-015-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_acquisition/test\_pubmed.py) for src/data\_acquisition/pubmed.py (mocking Biopython.Entrez calls using unittest.mock or pytest-mock):  
  * Test successful search and ID retrieval for a given keyword.  
  * Test successful fetching of XML content for a list of valid IDs.  
  * Test rate limiting implementation (e.g., verifying delays between calls).  
  * Test error handling for network issues, invalid queries, or empty results.  
  * Ensure Entrez.email is set.  
- [ ] **AIM2-ODIE-015-T2:** Create src/data\_acquisition/pubmed.py.  
- [ ] **AIM2-ODIE-015-T3:** Implement search\_pubmed(query: str, max\_results: int \= 100\) function using Bio.Entrez.esearch.  
- [ ] **AIM2-ODIE-015-T4:** Implement fetch\_pubmed\_xml(id\_list: list\[str\]) function using Bio.Entrez.efetch.  
- [ ] **AIM2-ODIE-015-T5:** Implement rate limiting (e.g., using time.sleep or a custom decorator) to comply with NCBI E-utilities guidelines (max 3 requests/sec without API key, 10 requests/sec with).  
- [ ] **AIM2-ODIE-015-T6:** Add robust error handling for Biopython.Entrez exceptions and API responses.  
- [ ] **AIM2-ODIE-015-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-016  
Description: PDF Text & Table Extraction Module: Develop a module (src/data\_acquisition/pdf\_extractor.py) using PyMuPDF and pdfplumber (or Camelot/Tabula-py) to extract text and tables from PDF documents.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-016-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_acquisition/test\_pdf\_extractor.py) for src/data\_acquisition/pdf\_extractor.py:  
  * Test text extraction from a simple, text-based PDF.  
  * Test table extraction from a PDF containing a clearly defined table.  
  * Test handling of multi-page PDFs for both text and table extraction.  
  * Error handling for non-existent PDF files.  
  * Error handling for corrupted or password-protected PDF files.  
- [ ] **AIM2-ODIE-016-T2:** Create src/data\_acquisition/pdf\_extractor.py.  
- [ ] **AIM2-ODIE-016-T3:** Implement extract\_text\_from\_pdf(file\_path: str) function using PyMuPDF (fitz.open().get\_text()) or pdfplumber (pdf.pages\[i\].extract\_text()).  
- [ ] **AIM2-ODIE-016-T4:** Implement extract\_tables\_from\_pdf(file\_path: str) function using pdfplumber (page.extract\_tables()) or Camelot/Tabula-py.  
- [ ] **AIM2-ODIE-016-T5:** Add error handling for PDF parsing issues specific to the chosen libraries.  
- [ ] **AIM2-ODIE-016-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-017  
Description: Scientific Journal Web Scraper (Metadata/Full Text): Develop a module (src/data\_acquisition/journal\_scraper.py) using paperscraper to scrape metadata and full-text (PDF/XML) from specified scientific journals, handling bot protection (User-Agent, throttling) and robots.txt.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-017-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_acquisition/test\_journal\_scraper.py) for src/data\_acquisition/journal\_scraper.py (mocking external requests and paperscraper calls):  
  * Test metadata scraping for a known journal article URL.  
  * Test full-text PDF/XML download for a known open-access article URL.  
  * Test User-Agent header setting.  
  * Test basic throttling (e.g., verifying time.sleep calls).  
  * Test robots.txt parsing and adherence (mock robots.txt file content).  
  * Error handling for HTTP errors (4xx, 5xx), connection issues, and scraping failures.  
- [ ] **AIM2-ODIE-017-T2:** Create src/data\_acquisition/journal\_scraper.py.  
- [ ] **AIM2-ODIE-017-T3:** Implement scrape\_journal\_metadata(journal\_name: str, query: str) function using paperscraper.  
- [ ] **AIM2-ODIE-017-T4:** Implement download\_journal\_fulltext(article\_url: str, output\_path: str) function using paperscraper or requests with appropriate headers.  
- [ ] **AIM2-ODIE-017-T5:** Implement check\_robots\_txt(url: str) to parse and respect robots.txt rules before scraping.  
- [ ] **AIM2-ODIE-017-T6:** Implement request throttling and User-Agent rotation strategies within the scraping functions.  
- [ ] **AIM2-ODIE-017-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-018  
Description: Text Cleaning & Preprocessing Module: Implement core text cleaning functionalities (src/text\_processing/cleaner.py): normalization (lowercase, special character removal), tokenization (spaCy/NLTK), duplicate removal (exact/fuzzy), stopword filtering, encoding standardization.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-018-T1:** **Develop Unit Tests:** Write unit tests (tests/text\_processing/test\_cleaner.py) for src/text\_processing/cleaner.py:  
  * Test normalize\_text for basic case conversion, whitespace stripping, and special character removal (e.g., HTML tags).  
  * Test tokenize\_text using spaCy or NLTK to ensure correct word/sentence segmentation.  
  * Test remove\_duplicates for exact duplicates and fuzzy matching with FuzzyWuzzy (e.g., "Compound A" vs "compound a").  
  * Test filter\_stopwords with a predefined list of common English and biomedical stopwords.  
  * Test standardize\_encoding for various input encodings.  
- [ ] **AIM2-ODIE-018-T2:** Create src/text\_processing/cleaner.py.  
- [ ] **AIM2-ODIE-018-T3:** Implement normalize\_text(text: str) function using regex and string methods.  
- [ ] **AIM2-ODIE-018-T4:** Implement tokenize\_text(text: str) function using spaCy or NLTK.  
- [ ] **AIM2-ODIE-018-T5:** Implement remove\_duplicates(text\_list: list\[str\], fuzzy\_threshold: int \= 90\) function using FuzzyWuzzy.  
- [ ] **AIM2-ODIE-018-T6:** Implement filter\_stopwords(tokens: list\[str\], custom\_stopwords\_list: list\[str\] \= None) function.  
- [ ] **AIM2-ODIE-018-T7:** Implement standardize\_encoding(text: bytes, target\_encoding: str \= 'utf-8') function.  
- [ ] **AIM2-ODIE-018-T8:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-019  
Description: Text Chunking Module for LLMs: Develop a module (src/text\_processing/chunker.py) for text chunking using strategies like fixed-size, sentence-based (NLTK/spaCy), and recursive character level (LangChain's RecursiveCharacterTextSplitter).  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-018  
Independent: No

- [ ] **AIM2-ODIE-019-T1:** **Develop Unit Tests:** Write unit tests (tests/text\_processing/test\_chunker.py) for src/text\_processing/chunker.py:  
  * Test chunk\_fixed\_size with various chunk sizes and overlaps, ensuring correct token counts.  
  * Test chunk\_by\_sentences using NLTK/spaCy sentence tokenizers, verifying chunks are complete sentences.  
  * Test chunk\_recursive\_char using LangChain's RecursiveCharacterTextSplitter with different separators and chunk sizes.  
  * Test handling of empty or very short texts.  
  * Verify that chunks maintain semantic coherence where possible (e.g., not splitting mid-word).  
- [ ] **AIM2-ODIE-019-T2:** Create src/text\_processing/chunker.py.  
- [ ] **AIM2-ODIE-019-T3:** Implement chunk\_fixed\_size(text: str, chunk\_size: int, chunk\_overlap: int) function.  
- [ ] **AIM2-ODIE-019-T4:** Implement chunk\_by\_sentences(text: str) function using NLTK or spaCy sentence tokenizers.  
- [ ] **AIM2-ODIE-019-T5:** Implement chunk\_recursive\_char(text: str, chunk\_size: int, chunk\_overlap: int, separators: list\[str\] \= None) function using LangChain's RecursiveCharacterTextSplitter.  
- [ ] **AIM2-ODIE-019-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-020  
Description: Named Entity Recognition (NER) Module (LLM-based): Implement NER functionality (src/llm\_extraction/ner.py) using LLM-IE or OntoGPT for zero-shot/few-shot extraction of specified entities (chemicals, species, traits, etc.). Focus on prompt engineering.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-019  
Independent: No

- [ ] **AIM2-ODIE-020-T1:** **Develop Unit Tests:** Write unit tests (tests/llm\_extraction/test\_ner.py) for src/llm\_extraction/ner.py (mocking LLM API calls using unittest.mock or pytest-mock):  
  * Test extract\_entities with a simple text and a predefined entity schema.  
  * Test zero-shot NER with a few example entity types.  
  * Test few-shot NER with provided examples in the prompt.  
  * Verify output format matches expected structured data (e.g., list of dictionaries with entity type, text, span).  
  * Error handling for LLM API failures, invalid responses, or rate limits.  
- [ ] **AIM2-ODIE-020-T2:** Create src/llm\_extraction/ner.py.  
- [ ] **AIM2-ODIE-020-T3:** Implement extract\_entities(text: str, entity\_schema: dict, llm\_model\_name: str, prompt\_template: str, few\_shot\_examples: list \= None) function using LLM-IE or OntoGPT.  
- [ ] **AIM2-ODIE-020-T4:** Define initial entity schemas for plant metabolites, species, plant anatomical structures, experimental conditions, molecular traits, and plant traits.  
- [ ] **AIM2-ODIE-020-T5:** Develop initial prompt templates for zero-shot and few-shot NER, focusing on clear instructions and output format.  
- [ ] **AIM2-ODIE-020-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-021  
Description: Relationship Extraction Module (LLM-based): Implement relationship extraction functionality (src/llm\_extraction/relations.py) using LLM-IE or OntoGPT to identify complex relationships between extracted entities. Focus on sophisticated prompt engineering and handling hierarchical relationships.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-020  
Independent: No

- [ ] **AIM2-ODIE-021-T1:** **Develop Unit Tests:** Write unit tests (tests/llm\_extraction/test\_relations.py) for src/llm\_extraction/relations.py (mocking LLM API calls):  
  * Test extract\_relationships with a simple text, a list of extracted entities, and a defined relationship schema.  
  * Test extraction of specific relationship types like "affects", "involved in", "upregulates".  
  * Test handling of hierarchical relationships (e.g., distinguishing "involved in" from "upregulates" based on context).  
  * Verify output format matches expected structured data (e.g., list of triples: (subject\_entity, relation\_type, object\_entity)).  
  * Error handling for LLM API failures or invalid responses.  
- [ ] **AIM2-ODIE-021-T2:** Create src/llm\_extraction/relations.py.  
- [ ] **AIM2-ODIE-021-T3:** Implement extract\_relationships(text: str, entities: list\[dict\], relationship\_schema: dict, llm\_model\_name: str, prompt\_template: str, few\_shot\_examples: list \= None) function using LLM-IE or OntoGPT.  
- [ ] **AIM2-ODIE-021-T4:** Define initial relationship schemas (e.g., Compound-Affects-Trait, Metabolite-InvolvedIn-BiologicalProcess).  
- [ ] **AIM2-ODIE-021-T5:** Develop sophisticated prompt templates for relationship extraction, including examples for hierarchical differentiation and contextual understanding.  
- [ ] **AIM2-ODIE-021-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-022  
Description: CLI for Literature Corpus Building: Create a command-line interface for the data acquisition modules, allowing users to download/scrape literature from various sources.  
Dependencies: AIM2-ODIE-015, AIM2-ODIE-016, AIM2-ODIE-017  
Independent: No

- [ ] **AIM2-ODIE-022-T1:** **Develop Integration Tests:** Write integration tests (tests/cli/test\_corpus\_cli.py) for the CLI:  
  * Test corpus pubmed-download \--query \<query\> \--output \<dir\> command (mocking pubmed.py calls).  
  * Test corpus pdf-extract \--input \<file\> \--output \<dir\> command (mocking pdf\_extractor.py calls).  
  * Test corpus journal-scrape \--url \<url\> \--output \<dir\> command (mocking journal\_scraper.py calls).  
  * Test invalid arguments and ensure proper error messages are displayed.  
- [ ] **AIM2-ODIE-022-T2:** Extend src/cli.py with a corpus subcommand using the chosen CLI framework.  
- [ ] **AIM2-ODIE-022-T3:** Implement corpus pubmed-download subcommand, calling functions from src/data\_acquisition/pubmed.py and handling output saving.  
- [ ] **AIM2-ODIE-022-T4:** Implement corpus pdf-extract subcommand, calling functions from src/data\_acquisition/pdf\_extractor.py and handling output saving.  
- [ ] **AIM2-ODIE-022-T5:** Implement corpus journal-scrape subcommand, calling functions from src/data\_acquisition/journal\_scraper.py and handling output saving.  
- [ ] **AIM2-ODIE-022-T6:** Add comprehensive help messages for all corpus commands and their arguments.  
- [ ] **AIM2-ODIE-022-T7:** **Conduct Integration Tests:** Run integration tests developed in T1.

Ticket ID: AIM2-ODIE-023  
Description: CLI for Text Preprocessing & LLM Extraction: Create a command-line interface for text cleaning, chunking, NER, and relationship extraction modules.  
Dependencies: AIM2-ODIE-018, AIM2-ODIE-019, AIM2-ODIE-020, AIM2-ODIE-021  
Independent: No

- [ ] **AIM2-ODIE-023-T1:** **Develop Integration Tests:** Write integration tests (tests/cli/test\_extraction\_cli.py) for the CLI:  
  * Test process clean \--input \<file\> \--output \<file\> command.  
  * Test process chunk \--input \<file\> \--output \<dir\> \--size \<int\> command.  
  * Test extract ner \--input \<file\> \--schema \<file\> \--output \<file\> command.  
  * Test extract relations \--input \<file\> \--entities \<file\> \--schema \<file\> \--output \<file\> command.  
  * Test invalid arguments and ensure proper error messages.  
- [ ] **AIM2-ODIE-023-T2:** Extend src/cli.py with process and extract subcommands.  
- [ ] **AIM2-ODIE-023-T3:** Implement process clean subcommand, calling functions from src/text\_processing/cleaner.py.  
- [ ] **AIM2-ODIE-023-T4:** Implement process chunk subcommand, calling functions from src/text\_processing/chunker.py.  
- [ ] **AIM2-ODIE-023-T5:** Implement extract ner subcommand, calling functions from src/llm\_extraction/ner.py.  
- [ ] **AIM2-ODIE-023-T6:** Implement extract relations subcommand, calling functions from src/llm\_extraction/relations.py.  
- [ ] **AIM2-ODIE-023-T7:** Add comprehensive help messages for all process and extract commands and their arguments.  
- [ ] **AIM2-ODIE-023-T8:** **Conduct Integration Tests:** Run integration tests developed in T1.

### **4\. Ontology Mapping and Post-processing**

Ticket ID: AIM2-ODIE-024  
Description: Entity-to-Ontology Mapping Module: Implement mapping of extracted entities to ontology terms using text2term. Support various mapping methods and minimum similarity scores.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-005, AIM2-ODIE-020  
Independent: No

- [ ] **AIM2-ODIE-024-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology\_mapping/test\_entity\_mapper.py) for src/ontology\_mapping/entity\_mapper.py:  
  * Test map\_entities\_to\_ontology with a list of extracted entity strings and a small, predefined test ontology.  
  * Test different text2term mapping methods (e.g., Mapper.TFIDF, Mapper.LEVENSHTEIN).  
  * Test min\_score filtering to ensure only high-confidence mappings are returned.  
  * Test mapping to specific term types (class, property).  
  * Test handling of unmapped terms (if incl\_unmapped is used).  
- [ ] **AIM2-ODIE-024-T2:** Create src/ontology\_mapping/entity\_mapper.py.  
- [ ] **AIM2-ODIE-024-T3:** Implement map\_entities\_to\_ontology(entities: list\[str\], ontology\_iri: str, mapping\_method: str \= 'tfidf', min\_score: float \= 0.3, term\_type: str \= 'class') function using text2term.map\_terms().  
- [ ] **AIM2-ODIE-024-T4:** Integrate Owlready2 loaded ontology for text2term's target\_ontology parameter if direct Owlready2 object passing is preferred over IRI.  
- [ ] **AIM2-ODIE-024-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-025  
Description: Relationship-to-Ontology Mapping Module: Implement mapping of extracted relationships to defined ontology properties using text2term or custom logic, ensuring semantic consistency.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-005, AIM2-ODIE-021, AIM2-ODIE-024  
Independent: No

- [ ] **AIM2-ODIE-025-T1:** **Develop Unit Tests:** Write unit tests (tests/ontology\_mapping/test\_relation\_mapper.py) for src/ontology\_mapping/relation\_mapper.py:  
  * Test map\_relationships\_to\_ontology with a list of extracted relationship triples (e.g., (subject\_text, relation\_text, object\_text)) and the loaded ontology.  
  * Test mapping of relation\_text to existing Owlready2.ObjectProperty instances.  
  * Test handling of relationships that do not have a direct match in the ontology.  
  * Ensure semantic consistency (e.g., verifying that mapped subjects/objects adhere to the domain/range of the mapped property).  
- [ ] **AIM2-ODIE-025-T2:** Create src/ontology\_mapping/relation\_mapper.py.  
- [ ] **AIM2-ODIE-025-T3:** Implement map\_relationships\_to\_ontology(relationships: list\[tuple\], ontology\_obj: Any) function. This function will likely involve iterating through relationships, attempting to map relation\_text to Owlready2.ObjectProperty instances, and potentially mapping subject/object texts to ontology entities using text2term (from AIM2-ODIE-024).  
- [ ] **AIM2-ODIE-025-T4:** Add logic to verify that mapped subjects and objects conform to the domain and range of the mapped ObjectProperty in the ontology.  
- [ ] **AIM2-ODIE-025-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-026  
Description: Entity Name Normalization Module: Develop a module (src/data\_quality/normalizer.py) for normalizing entity names (case, spacing, phrasing) using Python string methods and FuzzyWuzzy for fuzzy matching.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-026-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_quality/test\_normalizer.py) for src/data\_quality/normalizer.py:  
  * Test normalize\_name for basic case conversion (e.g., "King ARTHUR" \-\> "King Arthur"), extra space removal, and handling of specific words (e.g., "the").  
  * Test find\_fuzzy\_matches with a list of names and a query, verifying correct fuzzy matching using FuzzyWuzzy.fuzz.ratio, partial\_ratio, token\_sort\_ratio, token\_set\_ratio.  
  * Test edge cases like empty strings or lists.  
- [ ] **AIM2-ODIE-026-T2:** Create src/data\_quality/normalizer.py.  
- [ ] **AIM2-ODIE-026-T3:** Implement normalize\_name(name: str) function using string methods and potentially regex.  
- [ ] **AIM2-ODIE-026-T4:** Implement find\_fuzzy\_matches(query: str, candidates: list\[str\], threshold: int \= 80\) function using FuzzyWuzzy.process.extract.  
- [ ] **AIM2-ODIE-026-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-027  
Description: Fact Deduplication & Entity Resolution Module: Implement deduplication of redundant facts and entity resolution using dedupe or recordlinkage to consolidate unique entities.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-026  
Independent: No

- [ ] **AIM2-ODIE-027-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_quality/test\_deduplicator.py) for src/data\_quality/deduplicator.py:  
  * Test deduplicate\_entities with a simple list of dictionaries representing entities, including exact duplicates.  
  * Test deduplicate\_entities with records containing minor variations, using dedupe or recordlinkage to identify approximate matches.  
  * Test the output format (e.g., a list of unique, consolidated entities).  
  * Test handling of empty input lists.  
- [ ] **AIM2-ODIE-027-T2:** Create src/data\_quality/deduplicator.py.  
- [ ] **AIM2-ODIE-027-T3:** Implement deduplicate\_entities(records: list\[dict\], fields: list\[str\], settings\_file: str \= None, training\_file: str \= None) function using dedupe or recordlinkage. This will involve setting up the deduplication process (e.g., defining fields, training if dedupe is used).  
- [ ] **AIM2-ODIE-027-T4:** Integrate normalize\_name from AIM2-ODIE-026 as a preprocessing step for fields used in deduplication.  
- [ ] **AIM2-ODIE-027-T5:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-028  
Description: NCBI Taxonomy Integration & Filtering Module: Integrate multitax and ncbi-taxonomist to fetch, manage, and filter NCBI taxonomic information for robust species identification.  
Dependencies: AIM2-ODIE-002  
Independent: Yes

- [ ] **AIM2-ODIE-028-T1:** **Develop Unit Tests:** Write unit tests (tests/data\_quality/test\_taxonomy.py) for src/data\_quality/taxonomy.py (mocking external multitax/ncbi-taxonomist calls if they access external APIs):  
  * Test load\_ncbi\_taxonomy to ensure successful loading of taxonomy data.  
  * Test filter\_species\_by\_lineage to filter species based on a given taxonomic lineage (e.g., "Viridiplantae").  
  * Test get\_lineage\_for\_species to retrieve the full taxonomic lineage for a given species name or ID.  
  * Test handling of non-existent species or invalid IDs.  
- [ ] **AIM2-ODIE-028-T2:** Create src/data\_quality/taxonomy.py.  
- [ ] **AIM2-ODIE-028-T3:** Implement load\_ncbi\_taxonomy() function using multitax.NcbiTx() to load the NCBI taxonomy.  
- [ ] **AIM2-ODIE-028-T4:** Implement filter\_species\_by\_lineage(taxonomy\_obj, target\_lineage: str) function using multitax.filter() or ncbi-taxonomist's subtree command.  
- [ ] **AIM2-ODIE-028-T5:** Implement get\_lineage\_for\_species(taxonomy\_obj, species\_name\_or\_id: str) function using multitax or ncbi-taxonomist's resolve command.  
- [ ] **AIM2-ODIE-028-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-029  
Description: CLI for Ontology Mapping & Post-processing: Create a command-line interface for the ontology mapping, normalization, deduplication, and taxonomy integration modules.  
Dependencies: AIM2-ODIE-024, AIM2-ODIE-025, AIM2-ODIE-026, AIM2-ODIE-027, AIM2-ODIE-028  
Independent: No

- [ ] **AIM2-ODIE-029-T1:** **Develop Integration Tests:** Write integration tests (tests/cli/test\_postprocessing\_cli.py) for the CLI:  
  * Test map entities \--input \<file\> \--ontology \<url\> \--output \<file\> command.  
  * Test map relations \--input \<file\> \--ontology \<url\> \--output \<file\> command.  
  * Test clean normalize \--input \<file\> \--output \<file\> command.  
  * Test clean deduplicate \--input \<file\> \--output \<file\> command.  
  * Test taxonomy filter \--input \<file\> \--lineage \<lineage\> \--output \<file\> command.  
  * Test invalid arguments and ensure proper error messages.  
- [ ] **AIM2-ODIE-029-T2:** Extend src/cli.py with map and clean (subcommands for normalize, deduplicate), and taxonomy subcommands.  
- [ ] **AIM2-ODIE-029-T3:** Implement map entities subcommand, calling functions from src/ontology\_mapping/entity\_mapper.py.  
- [ ] **AIM2-ODIE-029-T4:** Implement map relations subcommand, calling functions from src/ontology\_mapping/relation\_mapper.py.  
- [ ] **AIM2-ODIE-029-T5:** Implement clean normalize subcommand, calling functions from src/data\_quality/normalizer.py.  
- [ ] **AIM2-ODIE-029-T6:** Implement clean deduplicate subcommand, calling functions from src/data\_quality/deduplicator.py.  
- [ ] **AIM2-ODIE-029-T7:** Implement taxonomy filter subcommand, calling functions from src/data\_quality/taxonomy.py.  
- [ ] **AIM2-ODIE-029-T8:** Add comprehensive help messages for all new commands and arguments.  
- [ ] **AIM2-ODIE-029-T9:** **Conduct Integration Tests:** Run integration tests developed in T1.

### **5\. Evaluation and Benchmarking**

Ticket ID: AIM2-ODIE-030  
Description: Gold Standard Test Set Creation Tool: Develop a simple CLI tool (or script) to facilitate manual annotation of a small set of papers (\~25) to create gold standard test sets for NER and relationship extraction. This tool should support defining entities/relationships and exporting annotations in a structured format.  
Dependencies: AIM2-ODIE-001  
Independent: Yes

- [ ] **AIM2-ODIE-030-T1:** **Develop Unit Tests:** Write unit tests (tests/evaluation/test\_gold\_standard\_tool.py) for src/evaluation/gold\_standard\_tool.py:  
  * Test load\_document\_for\_annotation to load a text file.  
  * Test add\_entity\_annotation to add an entity with type, text, and span.  
  * Test add\_relationship\_annotation to add a relationship between two entity IDs.  
  * Test export\_annotations to a JSON or CSV file and verify its structure.  
  * Test error handling for invalid input or annotation conflicts.  
- [ ] **AIM2-ODIE-030-T2:** Create src/evaluation/gold\_standard\_tool.py.  
- [ ] **AIM2-ODIE-030-T3:** Implement load\_document\_for\_annotation(file\_path: str) function.  
- [ ] **AIM2-ODIE-030-T4:** Implement add\_entity\_annotation(doc\_id: str, entity\_type: str, text: str, start\_char: int, end\_char: int) function to store annotations in memory.  
- [ ] **AIM2-ODIE-030-T5:** Implement add\_relationship\_annotation(doc\_id: str, subject\_id: str, relation\_type: str, object\_id: str) function.  
- [ ] **AIM2-ODIE-030-T6:** Implement export\_annotations(output\_file: str) function to save annotations in a structured format (e.g., JSON Lines).  
- [ ] **AIM2-ODIE-030-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-031  
Description: LLM Performance Benchmarking Module: Implement a module (src/evaluation/benchmarker.py) to benchmark LLM performance for NER and relationship extraction against gold standard datasets. Calculate precision, recall, and F1-scores.  
Dependencies: AIM2-ODIE-002, AIM2-ODIE-020, AIM2-ODIE-021, AIM2-ODIE-030  
Independent: No

- [ ] **AIM2-ODIE-031-T1:** **Develop Unit Tests:** Write unit tests (tests/evaluation/test\_benchmarker.py) for src/evaluation/benchmarker.py:  
  * Test calculate\_ner\_metrics with dummy predicted and gold standard entity lists, verifying correct precision, recall, F1 calculation.  
  * Test calculate\_relation\_metrics with dummy predicted and gold standard relationship lists, verifying correct precision, recall, F1 calculation.  
  * Test run\_benchmark with a small dummy gold standard dataset and mock LLM extraction calls.  
  * Test handling of empty inputs or no matches.  
- [ ] **AIM2-ODIE-031-T2:** Create src/evaluation/benchmarker.py.  
- [ ] **AIM2-ODIE-031-T3:** Implement calculate\_ner\_metrics(gold\_entities: list\[dict\], predicted\_entities: list\[dict\]) function.  
- [ ] **AIM2-ODIE-031-T4:** Implement calculate\_relation\_metrics(gold\_relations: list\[tuple\], predicted\_relations: list\[tuple\]) function.  
- [ ] **AIM2-ODIE-031-T5:** Implement run\_benchmark(gold\_standard\_data: list\[dict\], llm\_ner\_function, llm\_relation\_function) function that iterates through gold data, calls LLM extraction functions (AIM2-ODIE-020, \-021), and aggregates metrics.  
- [ ] **AIM2-ODIE-031-T6:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-032  
Description: Manual Curation & Feedback Loop Tool: Develop a CLI tool (or script) to enable manual review and correction of LLM-generated extractions. This tool should allow human experts to provide feedback and correct errors, which can then be used to refine prompts or models.  
Dependencies: AIM2-ODIE-020, AIM2-ODIE-021  
Independent: Yes

- [ ] **AIM2-ODIE-032-T1:** **Develop Unit Tests:** Write unit tests (tests/evaluation/test\_curation\_tool.py) for src/evaluation/curation\_tool.py:  
  * Test load\_llm\_output to load LLM-generated entities/relations from a file.  
  * Test display\_for\_review to ensure text and extracted items are presented clearly (mocking print statements).  
  * Test apply\_correction to modify an entity or relation.  
  * Test save\_curated\_output to a file and verify its structure.  
- [ ] **AIM2-ODIE-032-T2:** Create src/evaluation/curation\_tool.py.  
- [ ] **AIM2-ODIE-032-T3:** Implement load\_llm\_output(file\_path: str) function to load LLM-generated extractions.  
- [ ] **AIM2-ODIE-032-T4:** Implement display\_for\_review(text: str, entities: list\[dict\], relations: list\[tuple\]) function (CLI-based display).  
- [ ] **AIM2-ODIE-032-T5:** Implement apply\_correction(extracted\_data: dict, correction\_type: str, old\_value: Any, new\_value: Any) function to modify entities/relations.  
- [ ] **AIM2-ODIE-032-T6:** Implement save\_curated\_output(curated\_data: dict, output\_file: str) function.  
- [ ] **AIM2-ODIE-032-T7:** **Conduct Unit Tests:** Run unit tests developed in T1.

Ticket ID: AIM2-ODIE-033  
Description: CLI for Evaluation & Curation: Create a command-line interface for the benchmarking and manual curation tools.  
Dependencies: AIM2-ODIE-031, AIM2-ODIE-032  
Independent: No

- [ ] **AIM2-ODIE-033-T1:** **Develop Integration Tests:** Write integration tests (tests/cli/test\_evaluation\_cli.py) for the CLI:  
  * Test eval benchmark \--gold \<file\> \--predicted \<file\> command.  
  * Test eval curate \--input \<file\> \--output \<file\> command.  
  * Test invalid arguments and ensure proper error messages.  
- [ ] **AIM2-ODIE-033-T2:** Extend src/cli.py with an eval subcommand.  
- [ ] **AIM2-ODIE-033-T3:** Implement eval benchmark subcommand, calling functions from src/evaluation/benchmarker.py.  
- [ ] **AIM2-ODIE-033-T4:** Implement eval curate subcommand, calling functions from src/evaluation/curation\_tool.py.  
- [ ] **AIM2-ODIE-033-T5:** Add comprehensive help messages for all eval commands and their arguments.  
- [ ] **AIM2-ODIE-033-T6:** **Conduct Integration Tests:** Run integration tests developed in T1.