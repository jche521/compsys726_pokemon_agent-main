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

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
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
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
