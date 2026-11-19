"""Top-level vocabularies of the GDPval taxonomy (class level; nothing here is taken from a single test item)."""

SECTORS_OCCUPATIONS = {
    'Finance and Insurance': ['Customer Service Representatives', 'Financial Managers',
                              'Financial and Investment Analysts', 'Personal Financial Advisors',
                              'Securities, Commodities, and Financial Services Sales Agents'],
    'Government': ['Administrative Services Managers', 'Child, Family, and School Social Workers',
                   'Compliance Officers', 'First-Line Supervisors of Police and Detectives', 'Recreation Workers'],
    'Health Care and Social Assistance': ['First-Line Supervisors of Office and Administrative Support Workers',
                                          'Medical Secretaries and Administrative Assistants',
                                          'Medical and Health Services Managers', 'Nurse Practitioners',
                                          'Registered Nurses'],
    'Information': ['Audio and Video Technicians', 'Editors', 'Film and Video Editors',
                    'News Analysts, Reporters, and Journalists', 'Producers and Directors'],
    'Manufacturing': ['Buyers and Purchasing Agents', 'First-Line Supervisors of Production and Operating Workers',
                      'Industrial Engineers', 'Mechanical Engineers', 'Shipping, Receiving, and Inventory Clerks'],
    'Professional, Scientific, and Technical Services': ['Accountants and Auditors',
                                                         'Computer and Information Systems Managers', 'Lawyers',
                                                         'Project Management Specialists', 'Software Developers'],
    'Real Estate and Rental and Leasing': ['Concierges', 'Counter and Rental Clerks',
                                           'Property, Real Estate, and Community Association Managers',
                                           'Real Estate Brokers', 'Real Estate Sales Agents'],
    'Retail Trade': ['First-Line Supervisors of Retail Sales Workers', 'General and Operations Managers',
                     'Pharmacists', 'Private Detectives and Investigators'],
    'Wholesale Trade': ['First-Line Supervisors of Non-Retail Sales Workers', 'Order Clerks', 'Sales Managers',
                        'Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products',
                        'Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products'],
}
OCC_SECTOR = {o: s for s, os_ in SECTORS_OCCUPATIONS.items() for o in os_}
OCCUPATIONS = sorted(OCC_SECTOR)
assert len(OCCUPATIONS) == 44

TASK_TYPES = {
    'analysis_report': 'analyze information or data and write up findings and recommendations (memo, report, '
                       'assessment, briefing)',
    'spreadsheet_model': 'build or complete a quantitative workbook: budget, forecast, financial model, pricing, '
                         'calculations, scoring, variance analysis',
    'data_processing': 'transform, clean, consolidate or reconcile records into a structured output (tracker, '
                       'cleaned list, consolidated table, reconciliation)',
    'plan_schedule': 'a plan or schedule: project plan, staffing roster, timeline, itinerary, production or event plan',
    'form_template_checklist': 'a reusable blank form, template, checklist, log or tracker for others to fill in',
    'formal_document': 'formal, legal, regulatory or policy drafting: contract, agreement, will, policy, procedure, '
                       'compliance or clinical documentation',
    'correspondence': 'a letter, email, notice or written response to a specific person or party',
    'presentation': 'a slide deck',
    'editorial_content': 'articles, scripts, newsletters, press releases, marketing copy, or editing someone\'s copy',
    'technical_design': 'engineering or technical design work: calculations, specifications, architecture, drawings, '
                        'code',
    'training_guide': 'training or educational material, how-to guide, handout, SOP for a learner audience',
    'document_review': 'review, audit or critique a provided document and deliver comments, corrections or a redline',
    'proposal_pitch': 'a proposal, business case, bid, quote or sales pitch',
}

INPUT_ROLES = {
    'data_source': 'data or records the agent must read and compute/analyze from',
    'template_to_fill': 'a template, form or partially filled workbook/document the agent completes or updates',
    'reference_material': 'background material, guidelines, specs or instructions the deliverable must follow',
    'document_to_review': 'a document whose content the agent reviews, edits, critiques or rewrites',
    'media': 'image, audio or video',
}

PROVENANCE = {
    'private_business': 'internal/private business material: invoices, ledgers, HR or patient records, internal sales '
                        'or inventory sheets, rosters, internal memos, company templates, case files',
    'public_published': 'public-style material: regulations, laws, standards, filings, published forms, government '
                        'or industry reports, public guidance, manuals, articles, product datasheets',
    'mixed_other': 'neither clearly (e.g. a client brief, a draft written for this task) or media',
}
