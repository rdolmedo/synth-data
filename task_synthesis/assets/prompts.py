"""Prompts for sourcing: the query planner (per cell and catalog, no agent action), the shortlist (per spec, from
catalog metadata) and the fit check (per spec and fetched asset)."""

CLASS_TEXT = {
    'python-code': 'Python source code (a script, module or small package)', 'c-cpp-code': 'C/C++ source code',
    'other-code': 'source code', 'git-repo': 'a git repository with real commit history',
    'tabular-data': 'tabular data', 'text-data': 'text data', 'structured-data': 'structured data files',
    'running-service': 'the code or configuration of a network service or daemon (web API, queue, cache, mail, ...)',
    'ml-model': 'a small pretrained machine-learning model or tokenizer', 'binary': 'binaries or the source they are built from',
    'database': 'a database or database dump', 'media': 'media files', 'document': 'documents',
    'config': 'configuration files', 'archive': 'archives', 'scientific-data': 'scientific data files',
    'upstream-project': 'an upstream software project',
    # spec vocabulary v2 (#25) classes, bound since #118 (crypto-material is never bound: the scrubber redacts keys)
    'network-capture': 'network capture files or protocol logs (pcap/pcapng, HAR, flow records, Zeek or CAN logs)',
    'deployment-manifest': 'deployment or build manifests (Dockerfiles, compose files, Kubernetes manifests, Helm charts, '
                           'Terraform, Ansible, CI pipelines, systemd units)',
    'interface-schema': 'interface or schema definitions (OpenAPI, protobuf, GraphQL, JSON Schema, Avro, Thrift, XSD)',
    'software-package': 'software packages or their sources (Debian packages, Python wheels or sdists, npm tarballs, '
                        'gems, crates, R packages)',
}

SOURCE_TEXT = {
    'github-repo': 'GitHub repository search (matches repository names, descriptions, topics and READMEs; plain '
                   'keywords; the language filter is added for you)',
    'github-file': 'GitHub code search over file contents (plain keywords that would appear INSIDE such a file, '
                   'e.g. directives, field names, typical tokens; the file-type filter is added for you)',
    'zenodo': 'Zenodo research-data search (titles, descriptions, keywords of deposited datasets and papers)',
    'hf-dataset': 'Hugging Face dataset search (dataset names; short keywords work best)',
    'hf-model': 'Hugging Face model search (model names; short keywords like a task, architecture or domain)',
    'openml': 'OpenML dataset names and descriptions',
    'commons': 'Wikimedia Commons file search (file titles, descriptions, categories)',
    'arxiv': 'arXiv paper search (titles and abstracts)', 'europepmc': 'Europe PMC open-access article search',
    'wikipedia': 'Wikipedia article titles and text', 'gutenberg': 'Project Gutenberg catalog (titles, subjects, authors)',
    'rcsb': 'RCSB Protein Data Bank full-text search', 'ncbi': 'NCBI nucleotide/protein search (gene, organism, '
            'feature keywords)', 'pubchem': 'PubChem compound names or classes', 'massbank': 'MassBank spectra '
            '(compound names or classes, instrument types)', 'cod': 'Crystallography Open Database (mineral or '
            'compound names, formula elements)',
    'debian': 'Ubuntu package descriptions (small command-line tools and libraries)',
}

PLAN_SYSTEM = """You plan catalog searches that find real, public material for building realistic terminal tasks (the kind of job a skilled engineer, scientist or analyst does at a Linux shell). The material is found first; tasks are designed around it later, so do not think about what the task will be. Aim for breadth: different subfields, organizations, kinds of artifacts and eras. Avoid famous, tutorial or toy material (Iris, Titanic, MNIST, awesome-lists, homework, course exercises, boilerplate templates, hello-world projects)."""

PLAN_USER = """Material: {cls_text}{detail}
Domain: {domain}
Catalog: {source_text}

Write {k} search queries for this catalog that will each surface different, specific, real items of this material{domain_clause}. Each query is 1-4 plain keywords suited to the catalog's search engine (no boolean operators, quotes or filters). Answer with JSON only: {{"queries": ["...", ...]}}"""

SHORTLIST_SYSTEM = """You match real, public assets to task specifications for training AI agents on realistic terminal tasks. A task will be built around ONE asset as its main material (the anchor). Pick assets whose real content can carry a good task of the specification, offline, on 2 CPUs and 8 GB RAM without a GPU."""

SHORTLIST_USER = """# Specification
{spec}

# Candidates (catalog metadata only)
{cands}

Pick up to 3 candidates, most promising first, or none if none could anchor such a task. Prefer concrete, non-trivial real content in the right format and size; avoid empty templates, tutorials, famous showcase material and anything that needs the internet or a GPU at task time. Answer with JSON only: {{"picks": [candidate numbers], "why": "one sentence"}}"""

FIT_SYSTEM = """You decide whether a real, public asset can be the main material (the anchor) of a training task for AI agents, and if so which part of it to ship. A task author will later write the task around the asset: an instruction, companion files the specification asks for, and a setup script that can copy, convert, subset, compile or deliberately corrupt the asset. The agent works offline (package mirrors only) on 2 CPUs, 8 GB RAM, no GPU, with everything shipped.

Say fit only if a good task matching the specification can rest on this asset's REAL content:
- The specification's action and material must make sense with this asset. The task author may plant a defect into it, cut it down, or wrap it in authored code, but the asset must stay the core, not decoration next to invented data.
- Reverse-engineering tasks need something not answerable from memory: a widely documented format or famous program is a bad anchor unless the work is about this particular file's content.
- Tasks about broken, lost or hidden things need an asset a defect can be realistically planted into.
- No fit when the asset is too thin (empty template, stub, a few lines), unrelated to the domain or format, depends on external services or data it does not ship, or needs a GPU or more than a few GB to use.

When it fits, choose use_paths: the files or directories (relative paths as listed) to ship. Keep it to what a task needs: for code, the relevant module or package plus its README, build files and tests when useful (at most ~300 files, ideally far fewer); for data, the files of the right format. Use ["."] for the whole asset when all of it is needed and it is small."""

FIT_USER = """# Specification
{spec}

# Asset
Source: {source}; title: {title}; license: {license}
Description: {description}

{manifest}

Answer with JSON only: {{"fit": true or false, "reason": "one or two sentences", "use_paths": ["..."], "idea": "if fit: one or two sentences on how the asset could anchor the task"}}"""
