from pydantic import BaseModel, Field

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import json

from dotenv import load_dotenv
import os

load_dotenv()
# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



llm = OpenAI(model="gpt-4-turbo")


class Customer(BaseModel):
    """Data model for a customer behaviour."""
    Name: str = Field(..., description="Name of the customer")
    Type: str = Field(..., description="Type of the customer profile")
    Country: str = Field(..., description="Country")
    Age: str = Field(..., description="Age of the customers")
    AgeGroupWithSignificance: str = Field(..., description="Age group with significant presence")
    Gender: str = Field(..., description="Gender of the customer")
    IncomeLevel: str = Field(..., description="Income level of the customer")
    Residence: str = Field(..., description="Customer residences")
    Occupation: str = Field(..., description="Common occupations of the customer")
    VehicleOwnershipCount: str = Field(..., description="Number of vehicles owned by the customer")
    VehicleOwnershipPreferences: str = Field(..., description="Vehicle preferences of the customer")
    VehicleOwnershipDuration: str = Field(..., description="Ownership duration of the vehicles")
    PriceSensitivity: str = Field(..., description="Price sensitivity of the customers")
    SpendingMotivators: str = Field(..., description="Factors motivating customer spending")
    Values: str = Field(..., description="Values important to the customers")
    BrandLoyaltyLevel: str = Field(..., description="Level of brand loyalty among the customer")
    InterestInNewBrands: str = Field(..., description="Customer interest in new brands")
    PersonalInterests: str = Field(..., description="Personal interests of the customers")
    ValuesTradition: str = Field(..., description="Whether the customers value tradition")
    EngagementInitialStages: str = Field(..., description="Initial engagement stages preferred by the customers")
    TransactionPreference: str = Field(..., description="Transaction preferences of the customers")
    InformationSeeking: str = Field(..., description="Information seeking behavior of the customers")
    ServiceAppointmentPreferences: str = Field(..., description="Service appointment preferences")
    VehicleServicePickUpService: str = Field(..., description="Preference for vehicle pick-up service")
    LoanerVehicleRequirement: str = Field(..., description="Requirement for a loaner vehicle during service")
    TargetDemographic: str = Field(..., description="Target demographic for marketing")
    LuxuryExperienceWillingness: str = Field(..., description="Willingness for a luxury experience")
    DigitalEngagement: str = Field(..., description="Preferred digital platforms and engagement level")
    CommunicationPreferences: str = Field(..., description="Preferred methods of communication")
    PurchaseDecisionInfluencers: str = Field(..., description="Key influencers of purchase decisions")
    BrandPerception: str = Field(..., description="Perception of different brands")
    EnvironmentalConsciousness: str = Field(..., description="Awareness and concern for environmental issues")
    LoyaltyProgramAffiliation: str = Field(..., description="Participation in loyalty programs")
    FeedbackLikelihood: str = Field(..., description="Likelihood to provide feedback or reviews")
    SocialMediaActivity: str = Field(..., description="Level of activity on social media platforms")
    LeisureActivities: str = Field(..., description="Common leisure activities")
    ShoppingPreferences: str = Field(..., description="Preferred shopping channels and styles")
    TechnologyAdoptionRate: str = Field(..., description="Rate at which new technology is adopted")
    HealthAndWellnessConcerns: str = Field(..., description="Health and wellness concerns and priorities")
    EducationLevel: str = Field(..., description="Highest level of education attained")
    FamilyStatus: str = Field(..., description="Family composition and marital status")
    CulturalAffinities: str = Field(..., description="Cultural groups or activities with which the customer identifies")
    AccessibilityRequirements: str = Field(..., description="Any special accessibility requirements")
    PreferredPaymentMethods: str = Field(..., description="Favored methods for transactions")
    TravelFrequency: str = Field(..., description="Frequency of travel for leisure or business")
    MediaConsumptionHabits: str = Field(..., description="Preferred types of media and consumption habits")
    RiskTolerance: str = Field(..., description="Willingness to engage in risky activities or investments")
    CommunityInvolvement: str = Field(..., description="Level of involvement in local or online communities")
    PoliticalViews: str = Field(..., description="Political orientation or views")


def get_info_customer(text):
        
    prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert assistant for summarizing and extracting personality of a user from a text. If you don't find information, leave is as an empty string."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Here is the text: \n"
                    "------\n"
                    "{text}\n"
                    "------"
                ),
            ),
        ]
    )

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=Customer,
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    
    output = program(text=text)
    non_empty_attributes = {k: v for k, v in output.dict().items() if v}
    
    if len(non_empty_attributes) == 0:
        #
        return "No data", "No data"
    
    index = choose_class(non_empty_attributes)
    
    with open('people.json') as json_file:
        f"Given the following persona attributes:\n{non_empty_attributes}\n\n"
        data = json.load(json_file)
        
    
    return non_empty_attributes, data[index]
    
    #response = getPersonalizedResponse(text, non_empty_attributes, data[index])
    
    #return response    

def choose_class(persona):
    with open('people.json') as json_file:
        f"Given the following persona attributes:\n{persona}\n\n"
        data = json.load(json_file)
        # Prepare the prompt for the LLM
    scores = []
    for i in data:
        prompt = (
            f"Given the following persona attributes:\n{persona}\n\n"
            f"And the type defined in:\n{i}\n"
            "Provide a score, how similar they are from 1 to 100. Only give a number."
        )
        response = llm.complete(prompt).text

        scores.append(response)

    print(scores)
    max_score_index = scores.index(max(scores, key=int))

    return max_score_index

def getPersonalizedResponse(query, description, persona_class):
    prompt = (
        f"Given the persona:\n{description}\n\n"
        f"And the closest possible persona:\n{persona_class}\n\n"
        "Adjust your behaviour based on the conversation with available attributes. Your main target is to make the user buy electrical vehicles."
        f"Here is the user query: \n{query}\n"
    )
    return llm.complete(prompt).text


#print(get_info_customer("Hi, my name is Brian"))