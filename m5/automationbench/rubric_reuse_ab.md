You are checking a synthetic training task for contamination with one test task of AutomationBench, a benchmark of
workflow-automation agents. Each task is a request from a colleague plus the starting data of simulated business
apps (Gmail, Google Sheets, Slack, Salesforce, helpdesks, accounting apps, ...). The agent acts through the apps' APIs;
hidden checks reward the right changes and penalize changes to records that must stay untouched (the "traps").

The synthetic task was written without seeing the test task. Decide whether it nevertheless reuses the test task.

Verdicts:
- REUSE: the candidate is a variant of this specific test task. Any one of these is enough:
  - the same specific business scenario with the same core decision: the same workflow on the same kind of records,
    with the same deciding rule or the same trap construction (e.g. both: route the same kind of form submissions
    by the same region/size table, with an opted-out contact among them), even if every name and number differs;
  - the same scaffold: the same app set used in the same roles for the same process and the same trap(s);
  - copied distinctive specifics: several of the test task's people, companies, record titles, IDs, amounts or
    policy wording, or its request wording beyond generic phrases.
- WEAK_REUSE: a close relative: the same specific scenario but a clearly different deciding rule and trap, or the same
  rule and trap in a different process; or one or two copied distinctive specifics.
- NO_REUSE: only the genre is shared (e.g. both triage support tickets, both reconcile invoices, both email a list),
  or generic conventions (company.example.com addresses, common first names, standard app fields, boilerplate
  sentences such as "Include the names of affected entities in your message(s).").

Think about it, then end your answer with one line exactly of the form:
VERDICT: REUSE | WEAK_REUSE | NO_REUSE
