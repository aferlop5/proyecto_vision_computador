# Proyecto de Visión por Computador (SVM)

Este README es breve: requisitos y cómo ejecutar entrenamiento, evaluación y predicción con SVM.

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

## Instalación rápida
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Cómo ejecutar

Entrenamiento (SVM):
```bash
cd codigo
python3 main.py --entrenar --modelo svm
```

Evaluación (dataset completo):
```bash
cd codigo
python3 main.py --evaluar --modelo svm
```

Evaluación (dataset/test):
```bash
cd codigo
python3 main.py --evaluar-test --modelo svm
```

Predicción de una imagen:
```bash
cd codigo
python3 main.py --predecir ../dataset/tuercas/img_114.jpg --modelo svm --debug 1
```

## Notas
- Usa `python3` (no `python`).
- El modelo SVM se guarda en `modelos/svm_pipeline.joblib`.
- Los resultados y métricas se guardan en `resultados/`.
