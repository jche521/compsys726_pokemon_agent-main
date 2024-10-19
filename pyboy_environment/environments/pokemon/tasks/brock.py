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
        return [game_stats["badges"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = 0

        new_location = new_state["location"]
        new_frame = (new_state["location"]["map_id"], new_state["location"]["x"], new_state["location"]["y"])

        new_coord_reward = self.reward_new_coord(new_state)
        new_map_reward = self.reward_new_map(new_state)
        moving_reward = self.reward_moving(new_state)
        gain_xp_reward = self.reward_gain_xp(new_state)
        see_pokemon_reward = self.reward_see_new_pokemon(new_state)
        catch_pokemon_reward = self.reward_catch_new_pokemon(new_state)
        return  moving_reward + gain_xp_reward + see_pokemon_reward + catch_pokemon_reward
    
    

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 2000

    def reward_moving(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]

        old_x = self.prior_game_stats["location"]["x"]
        old_y = self.prior_game_stats["location"]["y"]

        if x != old_x or y != old_y:
            return 0.01
        return 0

    def reward_new_coord(self, new_state: dict[str, any]) -> float:
        x = new_state["location"]["x"]
        y = new_state["location"]["y"]
        if (x,y) not in self.visited_coord: # visit new coord
            self.visited_coord.add((x, y))
            return 0.1
        else:
            return 0
    
    def reward_new_map(self, new_state: dict[str, any]) -> float:
        map = new_state["location"]["map"]
        if map not in self.visited_map: #visit new map
            self.visited_map.add(map)
            self.visited_coord.clear() # clear the visited coord for new map
            print("NEW MAP")
            return 0.2
        return 0
    
    def reward_gain_xp(self, new_state: dict[str, any]) -> float:
        if sum(new_state["xp"]) > sum(self.prior_game_stats["xp"]):
            print("GAIN XP")
            return 0.5
        return 0
    
    def reward_catch_new_pokemon(self, new_state: dict[str, any]) -> float:
        if new_state["caught_pokemon"] > self.prior_game_stats["caught_pokemon"]:
            print("CATCH NEW POKEMON")
            return 0.8
        return 0

    def reward_see_new_pokemon(self, new_state: dict[str, any]) -> float:
        if new_state["seen_pokemon"] > self.prior_game_stats["seen_pokemon"]:
            print("SEE NEW POKEMON")
            return 0.5
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
