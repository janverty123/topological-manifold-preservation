# EXP-XXX: <Experiment Name>

**Status:** draft | active_debugging | validated | archived
**Config:** `configs/baseline/<config_file>.yaml`
**Related:** `EXPERIMENT_LOG.md#EXP-XXX`

## Objective

<What question is this experiment answering? One paragraph.>

## How to reproduce

```bash
python <entry_point_script>.py --config configs/baseline/<config_file>.yaml --method <finetune|ewc|tmp>
```

## Results summary

<Fill in after running -- key numbers, link to plots/ subfolder, and to
the raw .jsonl logs in logs/.>

| Method | Key metric 1 | Key metric 2 |
|---|---|---|
| finetune | | |
| ewc | | |
| tmp | | |

## Conclusion

<What did this experiment show? Does it confirm or reject a hypothesis
from OPEN_QUESTIONS.md?>

## Next steps

<What should happen after this experiment -- link to the next
EXPERIMENT_LOG.md entry if one was created as a result.>
