def is_nlos_miluv(tag_id) -> int:
    # NLOS
    if tag_id in [1, 3, 4]:
        return 1
    # LOS
    else:
        return 0


def get_obstacle_type_miluv(tag_id) -> int:
    tag_id_to_obstacle_map = {
        0: 0,
        1: 1,
        2: 0,
        3: 2,
        4: 3,
        5: 0,
    }
    return tag_id_to_obstacle_map[tag_id]
