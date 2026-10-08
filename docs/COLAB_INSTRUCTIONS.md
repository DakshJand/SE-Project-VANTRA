Colab OCR GPU sweep — manual instructions
==========================================

The browser relay is blocked (a stuck debugger attachment on the Colab tab prevents
any new managed tabs). The Colab notebook "Untitled3" in the user's Chrome already has
`!nvidia-smi` in its first cell and had a Connected runtime at last check.

Everything needed is prepared locally:

1. Notebook: vantra/vantra-api/scripts/colab_vantra_ocr_sweep.ipynb
   (upload via Colab: File > Upload notebook)
2. Data: vantra/data/realworld/colab_package.zip (141MB)
   - ft_indian/in_train, in_val, in_test (leak-free Indian splits, images+labels)
   - r3_recognizer.pt (production int'l checkpoint to init from)
   - intl_train_labels.txt (92k international labels; images NOT included — if you
     want the combined-corpus sweep, also upload the ft/train images or point the
     notebook's INTL_IMG_DIR at a Drive copy)
3. In Colab: Runtime > Change runtime type > T4 GPU. Verify with !nvidia-smi.
4. Upload colab_package.zip via the Colab file browser, then Run all.
5. Sweep: lr {1e-4, 5e-5, 2e-5} x epochs {12, 20} Indian-only + combined-from-r3
   (if int'l images present). ~15-30 min on T4.
6. Download best_checkpoints.zip when done (best val checkpoint + sweep_summary.csv).
7. Reference numbers (leak-free Indian test, n=312): deployed inB1_sub = 45.8% exact
   / 84.8% char; inA2 = 53.2%/87.5%; r3 = 30.4%/76.7%.
