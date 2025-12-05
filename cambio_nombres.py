#!/usr/bin/env python3
"""
Renombra imágenes del dataset de forma ordenada por clase.

- dataset/arandelas -> arandela_1.jpg, arandela_2.png, ...
- dataset/tornillos -> tornillo_1.jpg, ...
- dataset/tuercas   -> tuerca_1.jpg, ...
- dataset/test/(arandelas|tornillos|tuercas) si existen, igual que arriba.

Características:
- Mantiene la extensión original (.jpg/.png/etc.)
- Ordena por nombre antes de enumerar para estabilidad
- Evita colisiones: realiza un renombrado seguro
- Soporta "dry run" para visualizar cambios sin aplicarlos

Uso:
  python3 cambio_nombres.py              # renombra en ./dataset
  python3 cambio_nombres.py --dry-run    # solo muestra lo que haría
  python3 cambio_nombres.py --base ./dataset_alt  # directorio base alternativo
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_MAP = {
    "arandelas": "arandela",
    "tornillos": "tornillo",
    "tuercas": "tuerca",
}


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_images(folder: str) -> List[str]:
    try:
        names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except FileNotFoundError:
        return []
    imgs = [f for f in names if is_image(f)]
    return sorted(imgs)


def safe_rename(src: str, dst: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Renombra evitando colisiones. Devuelve (ok, mensaje)."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return True, f"Ya con nombre deseado: {src}"
    if os.path.exists(dst):
        return False, f"Destino ya existe, saltando: {dst}"
    if dry_run:
        return True, f"DRY-RUN: {src} -> {dst}"
    try:
        os.rename(src, dst)
        return True, f"Renombrado: {src} -> {dst}"
    except Exception as e:
        return False, f"Error renombrando '{src}' -> '{dst}': {e}"


def rename_in_folder(folder: str, prefix: str, dry_run: bool = False) -> List[str]:
    logs: List[str] = []
    imgs = list_images(folder)
    if not imgs:
        logs.append(f"Sin imágenes en {folder}")
        return logs

    # Plan de nombres destino
    planned = []
    for i, fname in enumerate(imgs, start=1):
        root, ext = os.path.splitext(fname)
        new_name = f"{prefix}_{i}{ext.lower()}"
        planned.append((fname, new_name))

    # Si algún destino entra en conflicto con un origen distinto, usar paso intermedio
    needs_temp = False
    existing = set(imgs)
    for src, dst in planned:
        if dst in existing and dst != src:
            needs_temp = True
            break

    # Paso intermedio (añadir sufijo temporal) si hay conflictos
    temp_suffix = ".tmp_ren"
    temp_map = []
    if needs_temp and not dry_run:
        for src, dst in planned:
            temp_name = f"{src}{temp_suffix}"
            ok, msg = safe_rename(os.path.join(folder, src), os.path.join(folder, temp_name), dry_run=dry_run)
            logs.append(msg)
            if ok:
                temp_map.append((temp_name, dst))
        # Segunda fase a destino final
        for tmp, dst in temp_map:
            ok, msg = safe_rename(os.path.join(folder, tmp), os.path.join(folder, dst), dry_run=dry_run)
            logs.append(msg)
        # Limpieza: si quedó algún tmp, intentar revertir
        for tmp, dst in temp_map:
            tmp_path = os.path.join(folder, tmp)
            if os.path.exists(tmp_path):
                # revertir
                orig = tmp.replace(temp_suffix, "")
                try:
                    os.rename(tmp_path, os.path.join(folder, orig))
                    logs.append(f"Revertido temporal: {tmp} -> {orig}")
                except Exception as e:
                    logs.append(f"No se pudo revertir temporal {tmp}: {e}")
    else:
        # Renombrado directo
        for src, dst in planned:
            ok, msg = safe_rename(os.path.join(folder, src), os.path.join(folder, dst), dry_run=dry_run)
            logs.append(msg)

    return logs


def process_base_dir(base_dir: str, dry_run: bool = False) -> None:
    # Carpetas principales
    for cls_dir, prefix in CLASS_MAP.items():
        folder = os.path.join(base_dir, cls_dir)
        logs = rename_in_folder(folder, prefix, dry_run=dry_run)
        for line in logs:
            print(line)

    # Test: si tiene subcarpetas por clase, procesarlas igual
    test_dir = os.path.join(base_dir, "test")
    if os.path.isdir(test_dir):
        for cls_dir, prefix in CLASS_MAP.items():
            folder = os.path.join(test_dir, cls_dir)
            if os.path.isdir(folder):
                logs = rename_in_folder(folder, prefix, dry_run=dry_run)
                for line in logs:
                    print(line)
        # Si test contiene imágenes sueltas, renombrarlas a test_1, test_2...
        test_imgs = list_images(test_dir)
        if test_imgs:
            logs = rename_in_folder(test_dir, "test", dry_run=dry_run)
            for line in logs:
                print(line)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Renombra imágenes del dataset por clase")
    parser.add_argument("--base", type=str, default="./dataset", help="Directorio base del dataset")
    parser.add_argument("--dry-run", action="store_true", help="Muestra cambios sin aplicarlos")
    args = parser.parse_args(argv)

    base_dir = os.path.abspath(args.base)
    if not os.path.isdir(base_dir):
        print(f"Directorio base no encontrado: {base_dir}")
        sys.exit(1)

    process_base_dir(base_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
