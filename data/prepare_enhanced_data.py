import pandas as pd
import numpy as np
import random

def create_enhanced_dataset():
    """Create an enhanced dataset with Shakespeare, classical literature, and complex texts"""
    
    # Simple texts (0.1-0.3)
    simple_texts = [
        "The dog runs in the park.",
        "I like to read books.",
        "The sun is bright today.",
        "She walks to school every day.",
        "The cat sat on the mat.",
        "Birds fly in the sky.",
        "We eat dinner together.",
        "The baby sleeps quietly.",
        "Flowers bloom in spring.",
        "The car drives fast.",
        "Children play outside.",
        "The book is on the table.",
        "Fish swim in water.",
        "The tree grows tall.",
        "We go to the store.",
        "The phone rings loudly.",
        "The door opens slowly.",
        "The clock ticks quietly.",
        "The wind blows gently.",
        "The rain falls softly."
    ]
    
    # Basic texts (0.3-0.5)
    basic_texts = [
        "The weather is nice today and we can go for a walk.",
        "I enjoy reading mystery novels in my free time.",
        "The restaurant serves delicious food from many countries.",
        "Learning a new language takes time and practice.",
        "The movie was entertaining with good special effects.",
        "My friend likes to travel and explore new places.",
        "The library has many books about different subjects.",
        "Cooking dinner together is fun and educational.",
        "The museum displays artifacts from ancient civilizations.",
        "Exercise helps keep your body healthy and strong.",
        "The concert featured music from various genres.",
        "Gardening requires patience and careful attention.",
        "The newspaper reports on local and world events.",
        "Photography captures moments and memories.",
        "The beach offers relaxation and beautiful scenery.",
        "Cooking requires following recipes and using ingredients.",
        "The park provides space for recreation and exercise.",
        "Reading improves vocabulary and comprehension skills.",
        "The zoo houses animals from around the world.",
        "The theater presents plays and musical performances."
    ]
    
    # Intermediate texts (0.5-0.7)
    intermediate_texts = [
        "The economic implications of globalization have fundamentally altered the way nations interact in the modern era.",
        "Environmental sustainability requires a comprehensive approach that balances human needs with ecological preservation.",
        "Technological innovation continues to reshape traditional industries and create new opportunities for growth.",
        "The psychological aspects of human behavior reveal complex patterns that influence decision-making processes.",
        "Cultural diversity enriches societies by fostering understanding and promoting creative problem-solving approaches.",
        "Scientific research methodology involves systematic investigation and rigorous analysis of empirical evidence.",
        "The philosophical foundations of democracy emphasize individual rights and collective responsibility.",
        "Educational reform initiatives aim to improve learning outcomes through evidence-based instructional strategies.",
        "Healthcare systems worldwide face challenges in providing accessible and affordable medical services.",
        "Urban planning considerations must address transportation, housing, and environmental concerns simultaneously.",
        "The historical development of constitutional law reflects evolving societal values and political priorities.",
        "International relations theory examines the complex dynamics between sovereign states and global institutions.",
        "The psychological impact of social media on adolescent development requires careful consideration and research.",
        "Economic policy decisions must balance short-term objectives with long-term sustainability goals.",
        "The ethical implications of artificial intelligence raise important questions about human autonomy and responsibility.",
        "Climate change adaptation strategies require coordinated efforts across multiple sectors and jurisdictions.",
        "The sociological factors influencing educational achievement include family background and community resources.",
        "Public health initiatives must address both individual behavior and systemic factors affecting population health.",
        "The technological revolution in communication has transformed how people connect and share information globally.",
        "The philosophical debate over free will versus determinism continues to challenge our understanding of human agency."
    ]
    
    # Advanced texts (0.7-0.85)
    advanced_texts = [
        "The epistemological foundations of modern science rest upon empirical observation and rigorous methodological frameworks that distinguish between correlation and causation.",
        "Constitutional jurisprudence necessitates careful consideration of both textual interpretation and historical context when addressing contemporary legal challenges.",
        "Socioeconomic disparities manifest themselves through multifaceted channels, perpetuating cycles of inequality across generations despite policy interventions.",
        "Metacognitive strategies enable learners to monitor and regulate their cognitive processes effectively, thereby enhancing educational outcomes.",
        "The intricate mechanisms underlying quantum entanglement continue to perplex even the most distinguished physicists in the field.",
        "Neuroplasticity research demonstrates the brain's remarkable capacity for adaptation and reorganization throughout the lifespan.",
        "The philosophical implications of consciousness studies challenge our fundamental assumptions about subjective experience and objective reality.",
        "Epigenetic mechanisms provide insights into how environmental factors influence gene expression without altering DNA sequences.",
        "The anthropological study of cultural evolution reveals complex patterns of social organization and technological development.",
        "Quantum computing algorithms leverage superposition and entanglement to solve computational problems intractable for classical systems.",
        "The phenomenological approach to psychological research emphasizes the subjective experience of consciousness and intentionality.",
        "Theoretical physics explores the fundamental nature of space, time, and matter through mathematical frameworks and experimental validation.",
        "The hermeneutic tradition in philosophy examines the interpretation of texts and the understanding of human experience.",
        "Cognitive neuroscience investigates the neural substrates underlying complex mental processes and behavioral outcomes.",
        "The sociological analysis of power structures reveals how institutional arrangements perpetuate social hierarchies and inequalities.",
        "The epistemological debate between rationalism and empiricism continues to influence contemporary approaches to knowledge acquisition.",
        "The ontological implications of artificial intelligence raise profound questions about the nature of consciousness and personhood.",
        "The methodological challenges in cross-cultural research require careful consideration of linguistic, conceptual, and contextual factors.",
        "The theoretical frameworks of postcolonial studies examine the lasting effects of imperialism on contemporary global relations.",
        "The interdisciplinary nature of environmental science necessitates integration of biological, physical, and social scientific perspectives."
    ]
    
    # Shakespeare and Classical Literature (0.85-0.95)
    shakespeare_texts = [
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them.",
        "The quality of mercy is not strained. It droppeth as the gentle rain from heaven upon the place beneath. It is twice blest: It blesseth him that gives and him that takes.",
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May, And summer's lease hath all too short a date.",
        "Now is the winter of our discontent made glorious summer by this sun of York; And all the clouds that lour'd upon our house in the deep bosom of the ocean buried.",
        "All the world's a stage, and all the men and women merely players: they have their exits and their entrances; and one man in his time plays many parts.",
        "Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him. The evil that men do lives after them; The good is oft interred with their bones.",
        "Double, double toil and trouble; Fire burn and cauldron bubble. Fillet of a fenny snake, In the cauldron boil and bake; Eye of newt and toe of frog, Wool of bat and tongue of dog.",
        "Out, damned spot! out, I say! One: two: why, then, 'tis time to do't. Hell is murky! Fie, my lord, fie! a soldier, and afeard? What need we fear who knows it, when none can call our power to account?",
        "Et tu, Brute? Then fall, Caesar! The Ides of March have come, but not gone. The conspirators' daggers have found their mark, and Rome's greatest leader lies bleeding.",
        "The course of true love never did run smooth. But, either it was different in blood, or else misgraffed in respect of years, or else it stood upon the choice of friends.",
        "Is this a dagger which I see before me, the handle toward my hand? Come, let me clutch thee. I have thee not, and yet I see thee still. Art thou not, fatal vision, sensible to feeling as to sight?",
        "What's in a name? That which we call a rose by any other name would smell as sweet. So Romeo would, were he not Romeo call'd, retain that dear perfection which he owes without that title.",
        "The fault, dear Brutus, is not in our stars, but in ourselves, that we are underlings. Men at some time are masters of their fates: The fault, dear Brutus, is not in our stars, but in ourselves.",
        "But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.",
        "Tomorrow, and tomorrow, and tomorrow, creeps in this petty pace from day to day to the last syllable of recorded time, and all our yesterdays have lighted fools the way to dusty death."
    ]
    
    # Expert/Philosophical texts (0.9-1.0)
    expert_texts = [
        "The phenomenological reduction, as conceived by Edmund Husserl, constitutes a methodological procedure whereby the natural attitude is suspended, enabling the transcendental ego to apprehend the essential structures of consciousness through eidetic intuition.",
        "The ontological argument for the existence of God, as formulated by Anselm of Canterbury, posits that the concept of a being than which nothing greater can be conceived necessarily entails its existence, for existence in reality is greater than existence in the understanding alone.",
        "The categorical imperative, as articulated by Immanuel Kant, demands that one act only according to that maxim whereby one can, at the same time, will that it should become a universal law, thereby establishing the foundation for deontological ethics.",
        "The dialectical materialism of Karl Marx posits that the material conditions of production determine the social, political, and intellectual life processes, creating a historical progression through class struggle and revolutionary transformation.",
        "The hermeneutic circle, as elucidated by Hans-Georg Gadamer, describes the interpretive process whereby understanding emerges through the dialectical interaction between the interpreter's pre-understanding and the text's horizon of meaning.",
        "The uncertainty principle, formulated by Werner Heisenberg, establishes that the more precisely the position of a particle is determined, the less precisely its momentum can be known, fundamentally challenging classical notions of causality and determinism.",
        "The linguistic relativity hypothesis, associated with Benjamin Lee Whorf, suggests that the structure of a language influences the way its speakers conceptualize and experience reality, thereby affecting cognitive processes and cultural worldviews.",
        "The existentialist philosophy of Jean-Paul Sartre emphasizes that existence precedes essence, meaning that human beings first exist and then define themselves through their choices and actions, thereby bearing absolute responsibility for their being.",
        "The structuralist approach to anthropology, as developed by Claude Lévi-Strauss, examines cultural phenomena as manifestations of underlying universal structures of human thought, revealing patterns that transcend specific historical and geographical contexts.",
        "The postmodern critique of metanarratives, articulated by Jean-François Lyotard, challenges the legitimacy of grand theoretical frameworks that claim to provide universal explanations, advocating instead for localized and contingent forms of knowledge."
    ]
    
    # Create dataset with appropriate complexity scores
    data = []
    
    # Add simple texts (0.1-0.3)
    for text in simple_texts:
        score = random.uniform(0.1, 0.3)
        data.append({'text': text, 'target': score})
    
    # Add basic texts (0.3-0.5)
    for text in basic_texts:
        score = random.uniform(0.3, 0.5)
        data.append({'text': text, 'target': score})
    
    # Add intermediate texts (0.5-0.7)
    for text in intermediate_texts:
        score = random.uniform(0.5, 0.7)
        data.append({'text': text, 'target': score})
    
    # Add advanced texts (0.7-0.85)
    for text in advanced_texts:
        score = random.uniform(0.7, 0.85)
        data.append({'text': text, 'target': score})
    
    # Add Shakespeare texts (0.85-0.95)
    for text in shakespeare_texts:
        score = random.uniform(0.85, 0.95)
        data.append({'text': text, 'target': score})
    
    # Add expert texts (0.9-1.0)
    for text in expert_texts:
        score = random.uniform(0.9, 1.0)
        data.append({'text': text, 'target': score})
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/enhanced_readability.csv', index=False)
    
    print(f"Enhanced dataset created with {len(df)} samples")
    print(f"Complexity score range: {df['target'].min():.3f} - {df['target'].max():.3f}")
    print(f"Mean complexity: {df['target'].mean():.3f}")
    print(f"Standard deviation: {df['target'].std():.3f}")
    
    # Print distribution
    print("\nComplexity Distribution:")
    print(f"Simple (0.1-0.3): {len(df[df['target'] < 0.3])} texts")
    print(f"Basic (0.3-0.5): {len(df[(df['target'] >= 0.3) & (df['target'] < 0.5)])} texts")
    print(f"Intermediate (0.5-0.7): {len(df[(df['target'] >= 0.5) & (df['target'] < 0.7)])} texts")
    print(f"Advanced (0.7-0.85): {len(df[(df['target'] >= 0.7) & (df['target'] < 0.85)])} texts")
    print(f"Shakespeare (0.85-0.95): {len(df[(df['target'] >= 0.85) & (df['target'] < 0.95)])} texts")
    print(f"Expert (0.9-1.0): {len(df[df['target'] >= 0.9])} texts")

if __name__ == "__main__":
    create_enhanced_dataset() 