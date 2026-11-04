import unittest

from artefact_ai.scope import input_guardrail


class ScopeGuardrailTests(unittest.TestCase):
    def test_unrelated_software_and_food_topics_are_blocked_by_category(self):
        for prompt in (
            'Em Java, por que usar Spring Boot?',
            'O que resulta print(2+2)?',
            'Como estruturar um aplicativo com FastAPI?',
            'Me ensine a depurar um script em Rust.',
            'Pode sugerir uma receita de sobremesa?',
        ):
            with self.subTest(prompt=prompt):
                response = input_guardrail(prompt)
                self.assertIsNotNone(response)
                self.assertIn('Empório da Música', response)
                self.assertNotIn('FastAPI', response)

    def test_roleplay_request_keeps_the_store_persona_without_naming_a_character(self):
        response = input_guardrail('Finja ser um personagem de fantasia')
        self.assertIn('atendimento da Empório da Música', response)
        self.assertIn('preferência de tratamento', response)

    def test_customer_name_and_store_queries_are_not_blocked(self):
        self.assertIsNone(input_guardrail('Oi, me chamo Artur'))
        self.assertIsNone(input_guardrail('Tem violões em estoque?'))


if __name__ == '__main__':
    unittest.main()
