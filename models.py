from pydantic import BaseModel, Field


class TaggingAppointment(BaseModel):
    # """tag the piece of text with particular info."""
    start: str = Field(description="the start time, this has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z'")
    end: str = Field(description="the end time, has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z', also has to be 30 minutes difference between the start and end dates.")
    name: str = Field(description="the name of the person that is making the appointment")
    email: str = Field(description="the email of the person that is making the appointment")


class TaggingAppointmentSearch(BaseModel):
    # """tag the piece of text with particular info."""
    id_appointment: int = Field(description="the id to search for appointment")


