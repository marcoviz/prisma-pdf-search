"""Chat de terminal; cada pergunta usa somente trechos do PDF selecionado."""
import sys
from search import search_prompt

def main():
    try:
        answer = search_prompt()
    except Exception as exc:
        print(f'Não foi possível iniciar: {type(exc).__name__}. Confira o .env e o banco.', file=sys.stderr)
        return 1
    print('Prisma PDF — digite uma pergunta ou sair.')
    while True:
        try:
            question = input('PERGUNTA: ').strip()
            if question.lower() in {'sair', 'exit', 'quit'}:
                break
            if not question:
                continue
            print(f'RESPOSTA: {answer(question)}\n')
        except (EOFError, KeyboardInterrupt):
            print('\nAté logo.')
            break
        except Exception as exc:
            print(f'Consulta não concluída ({type(exc).__name__}). Tente novamente.', file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())
