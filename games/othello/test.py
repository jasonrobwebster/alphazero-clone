import engine

eng = engine.OthelloEngine(6)
print(eng.board)

print(eng.legal_moves())

eng.move((2, 1))

print(eng.board)

print(eng.legal_moves())

eng.move((1, 1))

print(eng.board)

print(eng.legal_moves())