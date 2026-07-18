# Dataset Layout

Downloaded archives are stored under `ai_training/data/downloads` and extracted
datasets are stored under `ai_training/data/source`. Both folders are ignored by
Git except for `.gitkeep`.

Expected source examples:

```text
ai_training/data/source/
  Phở/
  Bánh mì/
  Cơm tấm/
  Bún bò Huế/
  Gỏi cuốn/
  Bánh xèo/
  Mì Quảng/
  Xôi xéo/
  Cháo lòng/
  Bún thịt nướng/
  Bún riêu/
  Hủ tiếu/
  Bánh cuốn/
  Bánh tráng nướng/
  Cao lầu/
```

Nested layouts are also supported, for example:

```text
ai_training/data/source/Vietnamese Foods/Images/Phở/
```

Folder names may be accented or unaccented, and may use spaces, hyphens, or
underscores. Unknown folders are reported and are not guessed.
