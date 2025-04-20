suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

deck = [(rank, suit) for suit in suits for rank in ranks]

red_cards = [card for card in deck if card[1] in ["Hearts", "Diamonds"]]
p_red = len(red_cards) / len(deck)

hearts = [card for card in red_cards if card[1] == "Hearts"]
p_heart_given_red = len(hearts) / len(red_cards)

face_cards = [card for card in deck if card[0] in ["Jack", "Queen", "King"]]
face_diamonds = [card for card in face_cards if card[1] == "Diamonds"]
p_diamond_given_face = len(face_diamonds) / len(face_cards)

face_spades = [card for card in face_cards if card[1] == "Spades"]
face_queens = [card for card in face_cards if card[0] == "Queen"]

#combining two lists into a bigger one and then removing duplicates by storing it into a set type
spade_or_queen = set(face_spades + face_queens)
p_spade_or_queen_given_face = len(spade_or_queen) / len(face_cards)

print(f"1. P(Red card) = {p_red}")
print(f"2. P(Heart | Red card) = {p_heart_given_red}")
print(f"3. P(Diamond | Face card) = {p_diamond_given_face}")
print(f"4. P(Spade or Queen | Face card) = {p_spade_or_queen_given_face}")
