
## Files:
- Annotation guidance (PDF)

- 10K annotations: CSV. Text is only kept for review purposes after which it will be removed.

## Annotations:

- Text: Tweet
- id: Tweet ID
- Date: Date of tweet
- agendas_a:_don't_vote: Do not vote in election attitude
- agendas_a:_protest/campaign:  Protest/campaign attitude (not analyzed in present paper)
- agendas_a:_share_information:  Share information attitude (not analyzed in present paper)
- agendas_a:_vote:  Vote in election attitude (not analyzed in present paper)
- agendas_a:_vote_against_e: Vote against [entity] attitude
- agendas_a:_vote_for_e: Vote for [entity] attitude
- agendas_b:_good_outcome_is_possible: Some good outcome is possible attitude (not analyzed in present paper)
- agendas_b:_e/g_is_immoral/harmful: [Entity] is immoral or harmful attitude
- agendas_b:_e/g_is_moral/beneficial: [Entity] is moral or beneficial attitude
- agendas_b:_election_process_fair: Election is fair attitude (not analyzed in present paper)
- agendas_b:_election_process_flawed/manipulated_by_e/g: Election is flawed attitude (not analyzed in present paper)
- agendas_b:_self/g_is_at_risk: Someone is at risk (not analyzed in present paper)
- agendas_na: Not applicable (treated as a "0" label)
- agendas_none: No attitude (treated as a "0" label)
- agendas_unannotatable: Cannot annotate (treated as a "0" label)
- concerns_character_of_e: Character of [entity] concern (can be discussing perceived good or bad aspects of character)
- concerns_democracy: Democracy concern (can be discussing perceived good or bad aspects of democracy)
- concerns_economy: Economy concern (can be discussing perceived good or bad aspects of economy)
- concerns_environment: Environment concern (can be discussing perceived good or bad aspects of environment)
- concerns_fake_news/misinfo: Misinformation concern (can be discussing perceived good or bad aspects of misinformation)
- concerns_immigration/refugees: Immigration concern (can be discussing perceived good or bad aspects of immigration)
- concerns_international_alliances: International alliances concern (can be discussing perceived good or bad aspects of international alliances)
- concerns_na: Not applicable concern (treated as a "0" label)
- concerns_national_identity_&_pride: National identity and pride concern (can be discussing perceived good or bad aspects of national identity); contrast with pride emotion, which is about being proud
- concerns_none: No concern (treated as a "0" label)
- concerns_relationship_w/_russia: Relationship with Russia concern (can be discussing perceived good or bad aspects of relationship with Russia)
- concerns_religion: Religion concern (can be discussing perceived good or bad aspects of religion)
- concerns_terrorism/counterterrorism: Terrorism or counter-terrorism concern (can be discussing perceived good --counterterrorism-- or bad -- some terrorism-based danger)
- concerns_unannotatable: Unannotatable concern (treated as a "0" label)
- emotions_admiration/love: Admiration or love emotion
- emotions_amusement: Amusement "emotion"
- emotions_anger/contempt/disgust: Anger, contempt, or disgust emotion
- emotions_embarrassment/guilt/sadness: Embarrassment, guilt, or sadness emotion
- emotions_fear/pessimism: Pessimism emotion
- emotions_joy/happiness: Joy or happiness emotion
- emotions_na: Not applicable emotion (treated as a "0" label)
- emotions_negative-other: Negative-other emotion
- emotions_none: No emotion expressed
- emotions_optimism/hope: Optimism or hope emotion
- emotions_positive-other: Positive-other emotion 
- emotions_pride: Pride emotion (individual, group, or national pride); contrast with pride concern, which is not necessarily about being proud
- emotions_unannotatable: 

## Number of evaluators in total
3 annotators per document per Attitude, concern, or emotion category minimum - 3 or more annotators focused exclusively on the Moral Framing labels, 3 or more on the Attitudes, etc. Each group worked on an identical dataset but were presented only with the labels of the category they were assigned, then all label data was combined in our final eval dataset. There were approximately 15 annotators in total. In some cases, more than 3 annotators labeled a given tweet for a given label category.

## Information about evaluators per tweet
All annotators were presented with all labels (there was no grouping). 1

## Exact questions given to evaluators to determine the emotions
In 1B, annotators were given a training document (labeling guidelines) and participated in a series of training and check-in meetings to ensure that they understood the tasks (and had an opportunity to ask clarifying questions) and were using a consistent labeling approach.

## Annotations were binary, multilabeling was possible
Annotators were instructed to label whether there was a given attitude, concern, or emotion in each tweet. Multiple and overlapping annotations (such as multiple emotions or multiple attitudes with a given tweet) were possible. Each time a given annotator placed 1 or more label(s) in a document, we considered that to be equivalent to a document-level binary 1 (yes, present) with respect to that label for that annotator, and lack of a label (or application of a categorically negative 'none of these' label) was treated as document-level binary 0 (for that category/document/annotator combination).
