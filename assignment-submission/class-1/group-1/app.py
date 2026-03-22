import importlib.util
import random
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
HYBRID_FILE = BASE_DIR / "hybrid-eliza-multi-final.py"


def _resolve_oppaword_paths() -> tuple[str, str]:
	script_path = BASE_DIR / "oppaWord" / "oppa_word.py"
	dict_path = BASE_DIR / "oppaWord" / "data" / "myg2p_mypos.dict"

	if not script_path.exists():
		raise RuntimeError(f"oppaWord script not found: {script_path}")
	if not dict_path.exists():
		raise RuntimeError(f"oppaWord dictionary not found: {dict_path}")

	return str(script_path), str(dict_path)


def _load_hybrid_module():
	spec = importlib.util.spec_from_file_location("hybrid_eliza_multi_final", HYBRID_FILE)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Cannot import module from {HYBRID_FILE}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _infer_lang_from_model_path(model_path: str) -> str:
	lower = model_path.lower()
	if "_en" in lower:
		return "en"
	return "mya"


@st.cache_resource
def load_eliza(model_path: str):
	module = _load_hybrid_module()
	lang = _infer_lang_from_model_path(model_path)
	init_kwargs = {
		"lang": lang,
		"model_path": model_path,
	}

	if lang == "mya":
		oppaword_script, oppaword_dict = _resolve_oppaword_paths()
		init_kwargs.update(
			{
				"tokenizer_name": "oppaword",
				"oppaword_script": oppaword_script,
				"oppaword_dict": oppaword_dict,
			}
		)

	eliza = module.HybridEliza(**init_kwargs)
	eliza.load_model()

	if eliza.model is None:
		raise RuntimeError(
			f"Model not found or failed to load from: {model_path}. "
			"Please provide a valid .pth path."
		)

	return {
		"module": module,
		"eliza": eliza,
		"lang": lang,
	}


def _ensure_chat_state(initial_message: str):
	if "messages" not in st.session_state:
		st.session_state.messages = [{"role": "assistant", "content": initial_message}]


def _reset_chat_with_greeting(greeting: str):
	st.session_state.messages = [{"role": "assistant", "content": greeting}]


def main():
	st.set_page_config(page_title="Hybrid ELIZA Chat", page_icon="💬", layout="centered")
	st.title("Hybrid ELIZA Chat")
	st.caption("Chat using a trained model checkpoint (.pth) only")

	model_default = str(BASE_DIR / "eliza_eq_mya.pth")
	model_path = st.text_input("Model path", value=model_default, help="Absolute or relative path to a .pth model file")

	model_file = Path(model_path).expanduser()
	if not model_file.is_absolute():
		model_file = (BASE_DIR / model_file).resolve()

	if not model_file.exists():
		st.error(f"Model file not found: {model_file}")
		return

	try:
		loaded = load_eliza(str(model_file))
	except Exception as exc:
		st.error(f"Failed to load model: {exc}")
		return

	eliza = loaded["eliza"]
	lang = loaded["lang"]
	scripts = loaded["module"].SCRIPTS

	initial_msg = random.choice(scripts[lang]["initials"])
	_ensure_chat_state(initial_msg)

	if st.button("Reset Chat"):
		_reset_chat_with_greeting(random.choice(scripts[lang]["initials"]))

	for msg in st.session_state.messages:
		with st.chat_message(msg["role"]):
			st.write(msg["content"])

	user_text = st.chat_input("Type your message...")
	if not user_text:
		return

	st.session_state.messages.append({"role": "user", "content": user_text})
	with st.chat_message("user"):
		st.write(user_text)

	response = eliza.rule_respond(user_text)
	emotion, score = eliza.get_eq(user_text)
	assistant_text = f"{response}\n\n[EQ Analysis] Predicted Emotion: {emotion} ({score:.2%})"

	st.session_state.messages.append({"role": "assistant", "content": assistant_text})
	with st.chat_message("assistant"):
		st.write(assistant_text)


if __name__ == "__main__":
	main()
