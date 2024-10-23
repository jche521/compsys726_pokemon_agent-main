from functools import cached_property

import cv2
import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc
import torch

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        self.visited_coord = set()
        self.visited_map = set()
        self.is_touching_grass = False
        self.attack_count = 0
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
        
        stats = np.array([
            # game_stats["badges"],            # Number of badges
            game_stats["location"]["x"],      # Agent's x-coordinate
            game_stats["location"]["y"],      # Agent's y-coordinate
            # self._is_grass_tile(),            # on grass boolean
            # game_stats["location"]["map_id"],      # Agent's y-coordinate
            (sum(game_stats["levels"])-6)/5,
            self.enemy_hp,
            self.attack_count
            # sum(game_stats["xp"]) - 220,            # Total XP
            # game_stats["seen_pokemon"],   # Number of Pokemon seen
            # game_stats["caught_pokemon"],   # Number of Pokemon seen
        ])
        
        # Grab the RGB frame using the existing grab_frame function
        frame = self.grab_frame(height=240, width=300)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_normalized = frame_gray  # Normalize pixel values to [0, 1]
        frame_normalized = frame_gray / 255.0
        frame_resized = cv2.resize(frame_normalized, (75, 60), interpolation=cv2.INTER_AREA).flatten()
        frame_tensor = torch.tensor(frame_resized, dtype=torch.float32)
        stats_tensor = torch.tensor(stats, dtype=torch.float32)
        combined_tensor = torch.cat((stats_tensor, frame_tensor))
        return combined_tensor  

    def _calculate_reward(self, new_state: dict) -> float:
        # REWARD
        new_coord_reward = self.reward_new_coord(new_state)
        touch_grass_reward = self.reward_touch_grass(new_state)
        new_map_reward = self.reward_new_map(new_state)
        gain_xp_reward = self.reward_gain_xp(new_state)
    
        attack_reward = 0
        if self.in_dialog:
            if self.enemy_hp != self.get_enemy_hp():
                attack_reward = self.reward_attack_pokemon(new_state)
                print(f"---FIGHTING REWARD: {attack_reward}---")
        
        reward = new_coord_reward + touch_grass_reward + new_map_reward + gain_xp_reward + attack_reward
        # + new_map_reward + gain_xp_reward + see_pokemon_reward + attack_reward
        
        # PENALTY
        not_moving_penalty = self.penalty_not_moving(new_state)

        # visit_old_map_penalty = self.penalty_back_to_old_map(new_state)
        penalty = not_moving_penalty

        # return reward
        return reward + penalty
    
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

    ######################
    ## REWARD FUNCTIONS ##
    ######################

    def reward_new_coord(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]
        if (x,y) not in self.visited_coord: # visit new coord
            self.visited_coord.add((x, y))
            return 1
        else:
            return 0
        
    def reward_new_map(self, new_state: dict[str, any]) -> float:
        map = new_state["location"]["map_id"]
        old_map = self.prior_game_stats["location"]["map_id"]

        if map not in self.visited_map: #visit new map
            map_name = new_state["location"]["map"]
            print(f"---NEW MAP: {map_name}---")
            self.visited_map.add(map)
            return 3
        return 0
    
    def reward_touch_grass(self, new_state: dict[str, any]) -> int:
        if self._is_grass_tile():
            return 0.5
        return 0
    
    def penalty_not_moving(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]

        old_x = self.prior_game_stats["location"]["x"]
        old_y = self.prior_game_stats["location"]["y"]

        if not self.in_dialog() and x == old_x and y == old_y:
            return -0.1
        return 0
    
    def reward_gain_xp(self, new_state: dict[str, any]) -> float:
        if sum(new_state["xp"]) > sum(self.prior_game_stats["xp"]):
            print("---GAIN XP---")
            return 20 #10
        return 0

    def reward_attack_pokemon(self, new_state: dict[str, any]) -> float:
        new_enemy_hp = self.get_enemy_hp()
        hp = self._read_party_hp()
        self_hp = sum(hp["current"]) / sum(hp["max"])

        if new_enemy_hp < self.enemy_hp:
            print(f"--ATTACK: self: {self_hp} enemy: new {new_enemy_hp}, old {self.enemy_hp}--")
            self.attack_count += 1
            self.enemy_hp = new_enemy_hp
            return 5 * (self.attack_count ** 2)
        
        if self.attack_count != 0:
            self.attack_count = 0
            return -5
        
        self.enemy_hp = new_enemy_hp
        return 0
    
    def penalty_near_obstacle(self, new_state: dict[str, any]) -> float:
        if self.is_near_wall_or_lake(new_state):
            return -0.5  # Penalize strongly for sticking to walls or lakes
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
    
    #############################
    ## original implementation ##
    #############################

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
        enemy_current_hp = self._read_m(0xCFE7) 
        enemy_max_hp = self._read_m(0xCFF5)

        if enemy_current_hp == 0:
            return 0
        return enemy_current_hp/enemy_max_hp
        
    def in_battle(self) -> bool:
        return self._read_m(0xD057) != 0x00
    
    def in_dialog(self) -> bool:
        screen: np.ndarray = self.pyboy.game_area()  # 383
        is_in_dialog = False

        # if 383 exists in the game area, then the agent is stuck in dialog
        return 383 in screen
    
    def is_near_wall_or_lake(self) -> bool:
        walkable_map = self._get_screen_walkable_matrix()

        for i in range(2,6):
            for j in range(2,6):
                if walkable_map[i][j] == 0:
                    return True

        return False