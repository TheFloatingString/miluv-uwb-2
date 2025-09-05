def is_nlos_miluv(tag_id) -> int:
    # NLOS
    if tag_id in [1, 3, 4]:
        return 1
    # LOS
    else:
        return 0
