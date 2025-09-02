.PHONY: enter-repl
enter-repl:
	julia --project=. --threads=auto -i main.jl