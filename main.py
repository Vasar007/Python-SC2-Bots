import random

import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *


# 165 iterations per minute.
class SimpleBot(sc2.BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 66
        self.MAX_GATEWAYS = 4
        self.MAX_STARGATES = 3
        self.MAX_TECH_BUILDS = 2
        self.MAX_NEXUSES = 4
        self.MAX_SUPPLY_CAP = 200

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()  # In sc2/bot_ai.py
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    async def build_workers(self):
        if len(self.units(PROBE)) < len(self.units(NEXUS)) * 22 and \
           len(self.units(PROBE)) < self.MAX_WORKERS and self.supply_left >= 1:
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
        if self.units(NEXUS).amount < self.MAX_NEXUSES and \
           self.can_afford(NEXUS):
            await self.expand_now()

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
        print(self.iteration / self.ITERATIONS_PER_MINUTE)

        if self.units(NEXUS).amount < 2:
            return

        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and \
               not self.units(CYBERNETICSCORE).exists:
                await self.build_tech_buildings(CYBERNETICSCORE, pylon)
            elif self.units(GATEWAY).amount < (self.iteration /
                                               self.ITERATIONS_PER_MINUTE):
                if self.can_afford(GATEWAY) and \
                   not self.already_pending(GATEWAY):
                    if not self.units(GATEWAY).amount >= self.MAX_GATEWAYS or \
                       not self.units(WARPGATE).amount:
                        await self.build(GATEWAY, near=pylon)

            if not self.units(FORGE).exists:
                await self.build_tech_buildings(FORGE, pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.units(STARGATE).amount < (self.iteration /
                                                  self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(STARGATE) and \
                       not self.already_pending(STARGATE) and \
                       self.units(STARGATE).amount < self.MAX_STARGATES:
                        await self.build(STARGATE, near=pylon)

            await self.build_high_tech_buildings(pylon)

    async def build_offensive_force(self):
        for gateway in self.units(GATEWAY).ready.noqueue:
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                if self.can_afford(STALKER) and self.supply_left >= 2:
                    await self.do(gateway.train(STALKER))

        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left >= 4:
                await self.do(stargate.train(VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {
            STALKER: [16, 5],
            VOIDRAY: [8, 3]
        }

        for UNIT in aggressive_units:
            if self.units(UNIT).amount >= aggressive_units[UNIT][0] and \
               self.units(UNIT).amount >= aggressive_units[UNIT][1]:
                for unit in self.units(UNIT).idle:
                    await self.do(unit.attack(self.find_target(self.state)))
            elif self.units(UNIT).amount >= aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for unit in self.units(UNIT).idle:
                        await self.do(unit.attack(
                            random.choice(self.known_enemy_units))
                        )


def main():
    run_game(maps.get("AbyssalReefLE"), [
             Bot(Race.Protoss, SimpleBot()),
             Computer(Race.Terran, Difficulty.Hard)
             ], realtime=True)

if __name__ == "__main__":
    main()
