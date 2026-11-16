import unittest

from lensplit.core.casting import (
    Actor,
    CastingData,
    Role,
    ShotCasting,
    parse_actor_csv,
)


class ActorTests(unittest.TestCase):
    def test_window_and_availability(self):
        actor = Actor("a1", "张三", "2026-11-01", "2026-11-05")
        self.assertTrue(actor.is_available("2026-11-01"))
        self.assertTrue(actor.is_available("2026-11-05"))
        self.assertFalse(actor.is_available("2026-11-06"))
        self.assertFalse(actor.is_available("2026-10-31"))
        self.assertEqual(actor.window_text(), "11-01 ~ 11-05")

    def test_empty_window_means_always_available(self):
        actor = Actor("a1", "李四")
        self.assertTrue(actor.is_available("2030-01-01"))
        self.assertEqual(actor.window_text(), "未填写档期")

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            Actor("a1", "张三", "2026-11-05", "2026-11-01")
        with self.assertRaises(ValueError):
            Actor("a1", "张三", "11/05/2026", "")
        with self.assertRaises(ValueError):
            Actor("a1", "   ")

    def test_set_availability_is_validated(self):
        actor = Actor("a1", "张三")
        actor.set_availability("2026-11-01", "2026-11-09")
        self.assertEqual(actor.available_to, "2026-11-09")
        with self.assertRaises(ValueError):
            actor.set_availability("2026-11-10", "2026-11-01")
        self.assertEqual(actor.available_from, "2026-11-01")


class CastingDataTests(unittest.TestCase):
    def setUp(self):
        self.casting = CastingData()
        self.zhang = self.casting.add_actor("张三", "2026-11-01", "2026-11-05")
        self.li = self.casting.add_actor("李四")
        self.hero = self.casting.add_role("主角", self.zhang.uuid)
        self.extra = self.casting.add_role("替身", self.li.uuid)
        self.unassigned = self.casting.add_role("路人")

    def test_duplicate_actor_in_one_shot_is_a_hard_clash(self):
        self.casting.set_shoot_date("g1", "2026-11-02")
        self.casting.set_roles("g1", [self.hero.uuid, self.extra.uuid])
        self.casting.assign_actor(self.extra.uuid, self.zhang.uuid)
        issues = self.casting.conflicts({"g1": "001"})
        errors = [issue for issue in issues if issue.is_error]
        self.assertEqual(len(errors), 1)
        self.assertIn("001", errors[0].message)
        self.assertIn("2 个角色", errors[0].message)

    def test_unavailable_actor_on_shoot_day_is_an_error(self):
        self.casting.set_shoot_date("g1", "2026-11-09")
        self.casting.set_roles("g1", [self.hero.uuid])
        issues = self.casting.conflicts({"g1": "001"})
        self.assertTrue(any("不含拍摄日 2026-11-09" in issue.message for issue in issues))
        self.assertTrue(any(issue.is_error for issue in issues))

    def test_unassigned_role_is_a_warning(self):
        self.casting.set_roles("g1", [self.unassigned.uuid])
        issues = self.casting.conflicts({"g1": "002"})
        self.assertEqual(len(issues), 1)
        self.assertFalse(issues[0].is_error)
        self.assertIn("尚未分配演员", issues[0].message)

    def test_actor_playing_two_roles_across_shots_same_day_warns(self):
        self.casting.set_shoot_date("g1", "2026-11-03")
        self.casting.set_shoot_date("g2", "2026-11-03")
        self.casting.set_roles("g1", [self.hero.uuid])
        self.casting.set_roles("g2", [self.extra.uuid])
        self.casting.assign_actor(self.extra.uuid, self.zhang.uuid)
        issues = self.casting.conflicts({"g1": "001", "g2": "002"})
        self.assertEqual(len(issues), 1)
        self.assertFalse(issues[0].is_error)
        self.assertIn("2 个不同角色", issues[0].message)
        self.assertIn("001、002", issues[0].message)

    def test_empty_shot_produces_no_issue(self):
        self.assertEqual(self.casting.conflicts(), [])

    def test_merge_shots_unions_roles_and_drops_old_ids(self):
        self.casting.set_character_count("g1", 1)
        self.casting.set_shoot_date("g1", "2026-11-02")
        self.casting.set_roles("g1", [self.hero.uuid])
        self.casting.set_roles("g2", [self.extra.uuid])
        self.casting.merge_shots(["g1", "g2"], "g9")
        self.assertNotIn("g1", self.casting.shots)
        self.assertNotIn("g2", self.casting.shots)
        merged = self.casting.shot("g9")
        self.assertEqual(merged.role_ids, [self.hero.uuid, self.extra.uuid])
        self.assertEqual(merged.shoot_date, "2026-11-02")
        self.assertEqual(merged.character_count, 2)

    def test_split_shot_copies_assignments_to_every_piece(self):
        self.casting.set_shoot_date("g1", "2026-11-02")
        self.casting.set_character_count("g1", 2)
        self.casting.set_roles("g1", [self.hero.uuid])
        self.casting.split_shot("g1", ["n1", "n2"])
        self.assertIsNone(self.casting.shot("g1"))
        self.assertEqual(self.casting.shot("n1").role_ids, [self.hero.uuid])
        self.assertEqual(self.casting.shot("n2").shoot_date, "2026-11-02")
        self.assertEqual(self.casting.shot("n2").character_count, 2)

    def test_removing_role_cleans_shot_assignments(self):
        self.casting.set_roles("g1", [self.hero.uuid, self.extra.uuid])
        self.casting.remove_role(self.hero.uuid)
        self.assertEqual(self.casting.shot("g1").role_ids, [self.extra.uuid])

    def test_removing_actor_unassigns_roles(self):
        self.casting.remove_actor(self.zhang.uuid)
        self.assertEqual(self.casting.role(self.hero.uuid).actor_uuid, "")

    def test_roundtrip_and_restore(self):
        self.casting.set_shoot_date("g1", "2026-11-02")
        self.casting.set_character_count("g1", 3)
        self.casting.set_roles("g1", [self.hero.uuid])
        snapshot = self.casting.snapshot()
        rebuilt = CastingData.from_dict(snapshot)
        self.assertEqual(rebuilt.to_dict(), snapshot)

        self.casting.remove_role(self.hero.uuid)
        self.casting.restore(snapshot)
        self.assertEqual(self.casting.shot("g1").role_ids, [self.hero.uuid])
        self.assertEqual(self.casting.role(self.hero.uuid).name, "主角")

    def test_invalid_dates_and_unknown_roles_are_rejected(self):
        with self.assertRaises(ValueError):
            self.casting.set_shoot_date("g1", "2026/11/02")
        with self.assertRaises(ValueError):
            self.casting.set_roles("g1", ["missing"])
        with self.assertRaises(ValueError):
            ShotCasting("g1", -1)

    def test_invalid_snapshot_is_rejected(self):
        with self.assertRaises(ValueError):
            CastingData(
                actors=[Actor("a1", "张三")],
                roles=[Role("r1", "主角", "a2")],
            )


class AvailabilitySyncTests(unittest.TestCase):
    def setUp(self):
        self.casting = CastingData()
        self.zhang = self.casting.add_actor("张三", "2026-11-01", "2026-11-05")
        self.hero = self.casting.add_role("主角", self.zhang.uuid)
        self.casting.set_roles("g1", [self.hero.uuid])

    def test_shot_day_outside_window_is_reported(self):
        self.casting.set_shoot_date("g1", "2026-11-09")
        issues = self.casting.check_group_schedule("g1")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "outside")
        self.assertIn("不含拍摄日 2026-11-09", issues[0].message)
        self.assertEqual(self.casting.check_actor_schedule(self.zhang.uuid), issues)

    def test_day_inside_window_has_no_issue(self):
        self.casting.set_shoot_date("g1", "2026-11-03")
        self.assertEqual(self.casting.check_group_schedule("g1"), [])

    def test_empty_window_is_reported_and_seeded_from_shot(self):
        actor = self.casting.add_actor("李四")
        role = self.casting.add_role("替身", actor.uuid)
        self.casting.set_roles("g2", [role.uuid])
        self.casting.set_shoot_date("g2", "2026-11-08")

        issues = self.casting.check_group_schedule("g2")
        self.assertEqual([issue.kind for issue in issues], ["empty"])

        seeded = self.casting.fill_missing_windows("g2")
        self.assertEqual(seeded, ["李四"])
        self.assertEqual(actor.available_from, "2026-11-08")
        self.assertEqual(actor.available_to, "2026-11-08")
        self.assertEqual(self.casting.check_group_schedule("g2"), [])

    def test_extend_issues_covers_every_shoot_day(self):
        self.casting.set_shoot_date("g1", "2026-11-09")
        self.casting.set_shoot_date("g2", "2026-10-28")
        self.casting.set_roles("g2", [self.hero.uuid])
        issues = self.casting.check_actor_schedule(self.zhang.uuid)
        self.assertEqual(len(issues), 2)
        changed = self.casting.extend_for_issues(issues)
        self.assertEqual(changed, 2)
        self.assertEqual(self.zhang.available_from, "2026-10-28")
        self.assertEqual(self.zhang.available_to, "2026-11-09")
        self.assertEqual(self.casting.check_actor_schedule(self.zhang.uuid), [])

    def test_clamping_moves_shot_day_into_window(self):
        self.casting.set_shoot_date("g1", "2026-11-09")
        issues = self.casting.check_group_schedule("g1")
        moved = self.casting.clamp_for_issues(issues)
        self.assertEqual(moved, [("g1", "2026-11-05")])
        self.assertEqual(self.casting.shot("g1").shoot_date, "2026-11-05")

        self.casting.set_shoot_date("g1", "2026-10-01")
        moved = self.casting.clamp_for_issues(self.casting.check_group_schedule("g1"))
        self.assertEqual(moved, [("g1", "2026-11-01")])

    def test_day_within_window_keeps_valid_days(self):
        self.assertEqual(
            self.casting.day_within_window(self.zhang.uuid, "2026-11-03"), "2026-11-03"
        )
        self.assertEqual(
            self.casting.day_within_window(self.zhang.uuid, ""), ""
        )

    def test_shots_for_actor_lists_planned_days(self):
        self.casting.set_shoot_date("g1", "2026-11-03")
        self.casting.set_roles("g2", [self.hero.uuid])
        self.casting.set_shoot_date("g2", "2026-11-01")
        self.assertEqual(
            self.casting.shots_for_actor(self.zhang.uuid),
            [("g2", "2026-11-01"), ("g1", "2026-11-03")],
        )
        self.assertEqual(self.casting.shoot_days(self.zhang.uuid), ["2026-11-01", "2026-11-03"])


class RoleSlotTests(unittest.TestCase):
    def setUp(self):
        self.casting = CastingData()
        self.zhang = self.casting.add_actor("张三", "2026-11-01", "2026-11-05")

    def test_person_count_creates_named_slots(self):
        self.casting.set_character_count("g1", 2)
        self.assertEqual(
            [slot.name for slot in self.casting.shot_slots("g1")], ["角色 1", "角色 2"]
        )
        self.casting.set_character_count("g1", 3)
        self.assertEqual(
            [slot.name for slot in self.casting.shot_slots("g1")],
            ["角色 1", "角色 2", "角色 3"],
        )
        self.casting.set_character_count("g1", 1)
        self.assertEqual([slot.name for slot in self.casting.shot_slots("g1")], ["角色 1"])

    def test_growing_keeps_existing_names_and_actors(self):
        self.casting.set_character_count("g1", 1)
        slot = self.casting.shot_slots("g1")[0]
        self.casting.rename_role(slot.uuid, "阿强")
        self.casting.assign_actor(slot.uuid, self.zhang.uuid)
        self.casting.set_character_count("g1", 3)
        slots = self.casting.shot_slots("g1")
        self.assertEqual([item.name for item in slots], ["阿强", "角色 2", "角色 3"])
        self.assertEqual(slots[0].actor_uuid, self.zhang.uuid)

    def test_unassigned_slots_are_not_reported_as_conflicts(self):
        self.casting.set_character_count("g1", 2)
        self.casting.set_shoot_date("g1", "2026-11-03")
        self.assertEqual(self.casting.conflicts({"g1": "001"}), [])

    def test_same_actor_in_two_slots_is_an_error(self):
        self.casting.set_character_count("g1", 2)
        for role in self.casting.shot_slots("g1"):
            self.casting.assign_actor(role.uuid, self.zhang.uuid)
        errors = [issue for issue in self.casting.conflicts({"g1": "001"}) if issue.is_error]
        self.assertEqual(len(errors), 1)
        self.assertIn("2 个角色", errors[0].message)

    def test_split_gives_each_piece_its_own_slots(self):
        self.casting.set_character_count("g1", 2)
        slot = self.casting.shot_slots("g1")[0]
        self.casting.assign_actor(slot.uuid, self.zhang.uuid)
        self.casting.split_shot("g1", ["n1", "n2"])
        first = self.casting.shot_slots("n1")
        second = self.casting.shot_slots("n2")
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertNotEqual(first[0].uuid, second[0].uuid)
        self.assertEqual(first[0].actor_uuid, self.zhang.uuid)
        self.assertEqual(second[0].actor_uuid, self.zhang.uuid)

    def test_merge_rehomes_slot_roles(self):
        self.casting.set_character_count("g1", 1)
        self.casting.assign_actor(self.casting.shot_slots("g1")[0].uuid, self.zhang.uuid)
        self.casting.set_character_count("g2", 1)
        self.casting.merge_shots(["g1", "g2"], "g9")
        slots = self.casting.shot_slots("g9")
        self.assertEqual(len(slots), 2)
        self.assertTrue(all(slot.shot_uuid == "g9" for slot in slots))
        self.assertEqual(self.casting.shot("g9").character_count, 2)


class ActorCsvImportTests(unittest.TestCase):
    def test_parse_and_import_with_header(self):
        text = (
            "\ufeff姓名,可拍摄开始,可拍摄结束,备注\n"
            "张三,2026-11-01,2026-11-05,主角\n"
            "李四,,,配角\n"
        )
        rows, errors = parse_actor_csv(text)
        self.assertEqual(errors, [])
        casting = CastingData()
        result = casting.import_actors(rows)
        self.assertEqual((result.added, result.updated), (2, 0))
        self.assertEqual(casting.actors[0].window_text(), "11-01 ~ 11-05")
        self.assertEqual(casting.actors[1].notes, "配角")

    def test_same_name_updates_existing_actor(self):
        casting = CastingData()
        casting.add_actor("张三", "2026-11-01", "2026-11-05")
        rows, _ = parse_actor_csv("姓名,可拍摄开始,可拍摄结束,备注\n张三,2026-11-02,2026-11-09,改档期\n")
        result = casting.import_actors(rows)
        self.assertEqual((result.added, result.updated), (0, 1))
        self.assertEqual(len(casting.actors), 1)
        self.assertEqual(casting.actors[0].available_to, "2026-11-09")
        self.assertEqual(casting.actors[0].notes, "改档期")

    def test_bad_rows_are_reported_not_raised(self):
        casting = CastingData()
        rows, errors = parse_actor_csv("姓名,可拍摄开始,可拍摄结束\n,2026-11-01,2026-11-02\n")
        self.assertEqual(len(errors), 1)
        result = casting.import_actors(rows)
        self.assertEqual(result.added, 0)

        rows, _ = parse_actor_csv("姓名,可拍摄开始,可拍摄结束\n李四,2026-11-09,2026-11-01\n")
        result = casting.import_actors(rows)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.added, 0)

    def test_missing_name_column_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_actor_csv("开始,结束\n2026-11-01,2026-11-02\n")


if __name__ == "__main__":
    unittest.main()
