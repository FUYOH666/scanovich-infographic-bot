"""FSM states for bot."""

from aiogram.fsm.state import State, StatesGroup


class GenerationStates(StatesGroup):
    """States for image generation flow."""

    IDLE = State()  # Waiting for commands
    WAITING_PHOTOS = State()  # Waiting for 1 product photo
    WAITING_BRIEF = State()  # Waiting for text or voice brief
    PROCESSING = State()  # Processing (ASR → LLM → Gemini)
    SHOW_RESULT = State()  # Showing result, waiting for actions

