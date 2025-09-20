import random

hangman_stages = ['''
  +---+
  |   |
      |
      |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
      |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
  |   |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|   |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\  |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\  |
 /    |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\  |
 / \  |
      |
=========''']

def hangman():
    words = ["hangman", "apple", "unique"]  # list of words to choose from
    word = random.choice(words)
    lives = len(hangman_stages)-1
    guessed = set()
    
    print("Hangman Game")
    print("_ " * len(word))

    while lives > 0:
        print(hangman_stages[6 - lives])
        progress = "".join(c if c in guessed else "_" for c in word)
        print("Word: " + " ".join(progress))

        if all(c in guessed for c in word):
            print("Congratulations! You guessed the word:", word)
            break

        g = input("Guess a letter: ").lower()
        if len(g) != 1 or not g.isalpha():
            print("Invalid input. Please enter a single alphabetic letter.")
            continue
        if g in guessed:
            print("You already guessed that letter.")
            continue

        guessed.add(g) 

        if g in word:
            print("Good guess!")
        else:
            lives -= 1
            print(f"Wrong guess! You have {lives} lives left.")

    else:
        print(hangman_stages[-1])
        print("Out of lives. The word was:", word)

hangman()


