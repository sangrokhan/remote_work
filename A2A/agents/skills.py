from python_a2a.models.agent import AgentSkill

# Define the Summarization Skill
summarize_skill = AgentSkill(
    id='summarize_files',
    name='Summarize Files',
    description='Reads a list of files and generates a summary of their content.',
    tags=['summarize', 'text-processing'],
    examples=['Summarize projects.csv and updates.txt'],
    # Note: AgentSkill in python-a2a uses input_modes, output_modes, not input_schema directly in __init__ usually,
    # but for this custom usage we can attach it or plain ignore if strictly typed.
    # Looking at the class definition, it accepts **kwargs implicitly via from_dict but strict init?
    # Class def: name, description, id, tags, examples, input_modes, output_modes.
    # It does NOT have input_schema field in __init__.
    # We will omit input_schema for now as it's not in the detected class definition.
)

# Define the Email Skill
email_skill = AgentSkill(
    id='send_email',
    name='Send Email',
    description='Sends an email with the provided details.',
    tags=['email', 'communication'],
    examples=['Send email to stakeholders about project update'],
)
