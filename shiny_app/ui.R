fluidPage(
  tabsetPanel(
    
    tabPanel("Summary", fluid = TRUE,
             sidebarLayout(
                            sidebarPanel(numericInput('living_space', 'Boston Condo Size (sqft)', 650, min = 300, max = 1750),
                                         numericInput('bathroom_number', 'Number of Bathrooms', 1, min = 1, max = 3),
                                         numericInput('bedroom_number', 'Number of Bedrooms', 1, min = 1, max = 3)),
               
                            mainPanel(fluidRow(column(7,  verbatimTextOutput('text1'))))
                          )
            ),
   
    tabPanel("Exploratory Data Analysis", fluid = TRUE,
             fluidRow(12, column(6,plotOutput("plot2")),
                          column(6,plotOutput("plot3")),
                          column(6,plotOutput("plot4")),
                          column(6,plotOutput("plot5")),
                          column(6,plotOutput("plot6")),
                          column(12,plotOutput("plot7")),
                          column(12,plotOutput("plot8")),
                          column(12,plotOutput("plot9"))
                     )

            ),
    
   tabPanel("About", htmlOutput("about_text"))
          )

)

