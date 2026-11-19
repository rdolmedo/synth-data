# Task lists

Every seed task that does not get GLM trajectories is listed in one of these files,
together with the reason.

| File | Why the tasks are excluded |
|---|---|
| `excluded_gpu.txt` | They need a GPU. All 8 B200s are serving GLM-5.3-Flash. |
| `excluded_solution_leak.txt` | Their Dockerfile copies the reference solution into the agent's image. |
| `excluded_needs_nonroot.txt` | The challenge relies on the agent being an unprivileged user, which the single-uid rootless runtime cannot provide. |
| `dropped_oracle.txt` | The environment cannot be made to pass its own oracle in this runtime, after the fixes recorded there. Written after the oracle sweep. |
| `not_investigated_security.txt` | Security-themed tasks whose oracle still fails and whose triage was stopped by the model provider's safety filter, so the cause was never investigated. They stay out until someone triages them by hand. |

`glm_tasks.txt` is the final list the rollouts were run on. The `glm_*batch*.txt` files are
the rollout batches in the order they were queued: each new batch lists the tasks whose
oracle newly passed after a round of runtime/task-prep fixes (`glm_v5_batch1.txt`: oracle-v5,
after the Docker exec-user/nsfake, prebuilt-image, proxy and TB1-conversion fixes).
