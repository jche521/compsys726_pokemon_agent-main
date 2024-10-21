from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        self.visited_coord = set()
        self.visited_map = set()
        # self.visited_location = set()

        # enemy stats
        self.enemy_hp = -1

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()
     
        return np.array([
            game_stats["badges"],            # Number of badges
            game_stats["location"]["x"],      # Agent's x-coordinate
            game_stats["location"]["y"],      # Agent's y-coordinate
            # game_stats["location"]["map_id"],      # Agent's y-coordinate
            # sum(game_stats["xp"]),            # Total XP
            # game_stats["seen_pokemon"],   # Number of Pokemon seen
            # game_stats["caught_pokemon"],   # Number of Pokemon seen
        ])

    def _calculate_reward(self, new_state: dict) -> float:

        # REWARD
        new_coord_reward = self.reward_new_coord(new_state)
        # new_map_reward = self.reward_new_map(new_state)
        # gain_xp_reward = self.reward_gain_xp(new_state)
        # see_pokemon_reward = self.reward_see_new_pokemon(new_state)
        # catch_pokemon_reward = self.reward_catch_new_pokemon(new_state)
        # attack_reward = self.reward_attack_pokemon(new_state)
        # touch_grass_reward = self.reward_touch_grass(new_state)

        reward = new_coord_reward 
        # + new_map_reward + gain_xp_reward + see_pokemon_reward 
        
        # PENALTY
        # visit_old_map_penalty = self.penalty_back_to_old_map(new_state)
        # not_moving_penalty = self.penalty_not_moving(new_state)
        # penalty = not_moving_penalty
        
        return reward
        # return reward + penalty

    
    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        done = game_stats["badges"] > self.prior_game_stats["badges"]
    
        if done:
            # Reset visited coordinates and maps
            self.visited_coord.clear()
            self.visited_map.clear()

        return done

    def _check_if_truncated(self, game_stats: dict) -> bool:
        truncated = self.steps >= 2048
    
        if truncated:
            # Reset visited coordinates and maps
            self.visited_coord.clear()
            self.visited_map.clear()

        return truncated
    
    def reward_new_coord(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]
        if (x,y) not in self.visited_coord: # visit new coord
            self.visited_coord.add((x, y))
            return 0.5
        else:
            return 0

    def penalty_not_moving(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]

        old_x = self.prior_game_stats["location"]["x"]
        old_y = self.prior_game_stats["location"]["y"]

        if not self.in_dialog() and x == old_x and y == old_y:
            return -0.1
        return 0

    def penalty_near_obstacle(self, new_state: dict[str, any]) -> float:
        if self.is_near_wall_or_lake(new_state):
            return -0.3  # Penalize strongly for sticking to walls or lakes
        return 0
    
    def reward_new_map(self, new_state: dict[str, any]) -> float:
        map = new_state["location"]["map_id"]
        old_map = self.prior_game_stats["location"]["map_id"]

        # if map not in self.visited_map: #visit new map
        #     self.visited_map.add(map)
        #     self.visited_coord.clear() # clear the visited coord for new map
        #     print("NEW MAP")
        #     return 0.2
        # return 0
        if map != old_map:
            if map in self.visited_map:
                print(f"Back to an old map: {map} applying penalty.")
                return -1
            else:
                print(f"New map visited: {map} giving reward.")
                self.visited_map.add(map)
                self.visited_coord.clear() # clear the visited coord for new map
                return 4
        return 0
    
    def reward_attack_pokemon(self, new_state: dict[str, any]) -> float:
        new_enemy_hp = self.get_enemy_hp()

        # Check if the enemy HP has decreased
        if new_enemy_hp < self.enemy_hp:
            self.enemy_hp = new_enemy_hp
            return 0.2
        
        self.enemy_hp = new_enemy_hp
        return 0
    
    def reward_touch_grass(self, new_state: dict[str, any]) -> int:
        if self._is_grass_tile():
            return 0.2
        return 0
    
    # def penalty_back_to_old_map(self, new_state: dict[str, any]) -> float:
    #     map = new_state["location"]["map"]
    #     old_map = self.prior_game_stats["location"]["map"]

    #     if map != old_map and map in self.visited_map:
    #         return -0.1
    #     return 0

    def reward_gain_xp(self, new_state: dict[str, any]) -> float:
        if sum(new_state["xp"]) > sum(self.prior_game_stats["xp"]):
            print("GAIN XP")
            return 0.8 #0.5
        return 0
    
    def reward_catch_new_pokemon(self, new_state: dict[str, any]) -> float:
        if new_state["caught_pokemon"] > self.prior_game_stats["caught_pokemon"]:
            print("CATCH NEW POKEMON")
            return 1 #0.8
        return 0

    def reward_see_new_pokemon(self, new_state: dict[str, any]) -> float:
        if new_state["seen_pokemon"] > self.prior_game_stats["seen_pokemon"]:
            print("SEE NEW POKEMON")
            return 0.7 #0.5
        return 0
    
    # original implementation
    def _caught_reward(self, new_state: dict[str, any]) -> int:
        return new_state["caught_pokemon"] - self.prior_game_stats["caught_pokemon"]

    def _seen_reward(self, new_state: dict[str, any]) -> int:
        return new_state["seen_pokemon"] - self.prior_game_stats["seen_pokemon"]

    def _health_reward(self, new_state: dict[str, any]) -> int:
        return sum(new_state["hp"]["current"]) - sum(
            self.prior_game_stats["hp"]["current"]
        )

    def _xp_reward(self, new_state: dict[str, any]) -> int:
        return sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])

    ######################
    ## HELPER FUNCTIONS ##
    ######################
    def get_enemy_hp(self) -> None:
        return self._read_m(0xCFE7)
        

    def in_battle(self) -> bool:
        return self._read_m(0xD057) != 0x00
    
    def in_dialog(self) -> bool:
        screen: np.ndarray = self.pyboy.game_area()  # 383
        is_in_dialog = False

        # if 383 exists in the game area, then the agent is stuck in dialog
        return 383 in screen
    
    def is_near_wall_or_lake(self, x, y) -> bool:
        walkable_map = self._get_screen_walkable_matrix()

        return walkable_map[y + 1][x] == 0 or walkable_map[y - 1][x] == 0  or walkable_map[y][x+ 1] == 0 or walkable_map[y][x - 1] == 0
