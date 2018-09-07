import math
import os
import random
import time

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import *

# import keras

import cv2
import numpy as np


os.environ["SC2PATH"] = "D:/Program Files (x86)/Blizzard Games/StarCraft II"
HEADLESS = False


# 165 iterations per minute.
class CleverBot(sc2.BotAI):

    def __init__(self, use_model=False, save_train_data=False, debug=False):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 66
        self.MAX_GATEWAYS = 8
        self.MAX_STARGATES = 8
        self.MAX_TECH_BUILDS = 2
        self.MAX_NEXUSES = 4
        self.MAX_SUPPLY_CAP = 200

        self.use_model = use_model
        self.save_train_data = save_train_data
        self.debug = debug

        # DICT {UNIT_ID: LOCATION}
        # Every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}
        self.expand_dis_dir = {}
        self.ordered_exp_distances = {}
        self.do_something_after = 0
        self.train_data = []
        self.time_ = 0.0

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model(
                "models/BasicCNN-30-epochs-0.0001-LR-4.2"
            )

    def on_end(self, game_result):
        print("--- on_end called ---")
        print(game_result, self.use_model)

        with open("results.log", "a") as f:
            if self.use_model:
                f.write(f"Model {game_result}\n")
            else:
                f.write(f"Random {game_result}\n")
                if self.save_train_data and game_result == Result.Victory:
                    np.save(f"train_data/{str(int(time.time()))}.npy",
                            np.array(self.train_data))

    async def on_step(self, iteration):
        # self.iteration = iteration -> No more used!
        self.time_ = (self.state.game_loop / 22.4) / 60
        await self.build_scout()
        await self.scout()
        await self.distribute_workers()  # In sc2/bot_ai.py
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def build_scout(self):
        if self.units(OBSERVER).amount < math.floor(self.time_ / 3):
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                print(self.units(OBSERVER).amount, self.time_ / 3)
                if self.can_afford(OBSERVER) and self.supply_left >= 1:
                    await self.do(rf.train(OBSERVER))

    async def scout(self):
        # {DISTANCE_TO_ENEMY_START: EXPANSIONLOC}
        self.expand_dis_dir = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(
                self.enemy_start_locations[0]
            )
            if self.debug:
                print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        # Not need in Python >= 3.7!
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]

        # Removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]
        # End removing of scouts that are dead now.

        if self.units(ROBOTICSFACILITY).ready.amount == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 10

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if self.units(unit_type).idle.amount > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[dist]
                                # DICT {UNIT_ID: LOCATION}
                                active_locations = [self.scouts_and_spots[k]
                                                    for k in
                                                    self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in \
                                               self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                print(e)

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(
                        self.scouts_and_spots[obs.tag]
                    )))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1],
                              self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        draw_dict = {
            NEXUS: [15, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            PROBE: [1, (55, 200, 0)],
            ASSIMILATOR: [2, (55, 200, 0)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            STARGATE: [5, (255, 0, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],

            VOIDRAY: [3, (255, 100, 0)],
            # OBSERVER: [3, (255, 255, 255)],
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])),
                           draw_dict[unit_type][0], draw_dict[unit_type][1],
                           -1)

        main_base_names = ["nexus", "commandcenter", "orbitalcommand",
                           "planetaryfortress", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5,
                           (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15,
                           (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe", "scv", "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1,
                               (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3,
                               (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1,
                       (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = self.units(VOIDRAY).amount / (self.supply_cap -
                                                        self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19),
                 (250, 250, 200), 3)  # Worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15),
                 (220, 200, 200), 3)  # Plausible supply (supply / 200.0)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11),
                 (150, 150, 150), 3)  # Population ratio (supply_left / supply)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7),
                 (210, 200, 0), 3)  # Gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3),
                 (0, 255, 25), 3)  # Minerals minerals / 1500

        # Flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow("Intel", resized)
            cv2.waitKey(1)

    async def build_workers(self):
        if self.units(PROBE).amount < self.units(NEXUS).amount * 22 and \
           self.units(PROBE).amount < self.MAX_WORKERS and \
           self.supply_left >= 1:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON) and \
                   self.supply_cap < self.MAX_SUPPLY_CAP:
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0,
                                                           vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def expand(self):
        try:
            if self.units(NEXUS).amount < self.time_ and \
               self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(e)

    async def build_tech_buildings(self, name, pylon):
        if self.can_afford(name) and \
           not self.already_pending(name) and \
           self.units(name).amount <= self.MAX_TECH_BUILDS:
            await self.build(name, near=pylon)

    async def build_high_tech_buildings(self, pylon):
        if self.units(CYBERNETICSCORE).ready.exists and \
           self.units(STARGATE).ready.exists:
            if not self.units(TWILIGHTCOUNCIL).ready.exists:
                if self.can_afford(TWILIGHTCOUNCIL) and \
                   not self.already_pending(TWILIGHTCOUNCIL):
                    await self.build(TWILIGHTCOUNCIL, near=pylon)
            else:
                if not self.units(FLEETBEACON).ready.exists and \
                   self.can_afford(FLEETBEACON) and \
                   not self.already_pending(FLEETBEACON):
                    await self.build(FLEETBEACON, near=pylon)

    async def offensive_force_buildings(self):
        if self.debug:
            print("Time:", time_)

        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists and \
               not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and \
                   not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif self.units(GATEWAY).amount < 1:
                if self.can_afford(GATEWAY) and \
                   not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.units(ROBOTICSFACILITY).amount < 1:
                    if self.can_afford(ROBOTICSFACILITY) and \
                       not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.units(STARGATE).amount < self.time_ and \
                   self.units(STARGATE).amount < self.MAX_STARGATES:
                    if self.can_afford(STARGATE) and \
                       not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left >= 4:
                await self.do(stargate.train(VOIDRAY))

    def find_target(self, state):
        if self.known_enemy_units.amount > 0:
            return random.choice(self.known_enemy_units)
        elif self.known_enemy_structures.amount > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        if self.units(VOIDRAY).idle.amount > 0:
            target = False
            if self.time_ > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict(
                        [self.flipped.reshape([-1, 176, 200, 3])]
                    )
                    choice = np.argmax(prediction[0])
                    print("prediction: ", choice)

                    choice_dict = {
                        0: "No Attack!",
                        1: "Attack close to our nexus!",
                        2: "Attack Enemy Structure!",
                        3: "Attack Eneemy Start!"
                    }

                    print(f"Choice #{choice}: {choice_dict[choice]}")

                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no_attack
                    wait = random.randrange(7, 100) / 100
                    self.do_something_after = self.time_ + wait
                elif choice == 1:
                    # attack_unit_closest_nexus
                    if self.known_enemy_units.amount > 0:
                        target = self.known_enemy_units.closest_to(
                            random.choice(self.units(NEXUS))
                        )
                elif choice == 2:
                    # attack_enemy_structures
                    if self.known_enemy_structures.amount > 0:
                        target = random.choice(self.known_enemy_structures)
                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y, self.flipped])


def main():
    run_game(maps.get("AbyssalReefLE"), [
                 Bot(Race.Protoss, CleverBot()),
                 Computer(Race.Terran, Difficulty.Medium)
             ], realtime=False)


def test_model():
    for i in range(100):
        run_game(maps.get("AbyssalReefLE"), [
                     Bot(Race.Protoss, CleverBot(use_model=True)),
                     Computer(Race.Protoss, Difficulty.Medium),
                 ], realtime=False)


if __name__ == "__main__":
    main()
