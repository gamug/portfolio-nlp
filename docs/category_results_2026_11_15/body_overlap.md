# Bodies shared by articles and the folds of the selected run (2026-11-15)

`uv run scripts/category_body_overlap_2026_11_15.py --source-db <urls.db>`; the folds are the committed ones.

* articles: 1049; distinct bodies (`triage.body_hash`): 1048
* bodies that are the text of more than one article: 1
* of those, in more than one fold: 1 (articles [116568, 439320])
* rows in them: 16 of 1216 (1.3%); accuracy of the committed predictions on them 1.000
* accuracy on all rows 0.8783; without those rows (1200) 0.8767
