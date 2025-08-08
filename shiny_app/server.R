function(input, output, session) {
 
  #price_pred = (predict(lr, inputdata, level=.95, interval="confidence", response = TRUE))
  
  #output$table1 <- renderTable(inputdata())
  input_data <- reactive({
    data.frame(
      living_space = input$living_space,
      bedroom_number = input$bedroom_number,
      bathroom_number = input$bathroom_number
    )
  })

  output$text1 <- renderText({
    # Extracting the data frame from the reactive expression
    user_input <- input_data()

    # Predicting the price
    price_pred = predict(lr, user_input, level = 0.95, interval = "confidence", response = TRUE)

    # Returning the result
    paste("Predicted Price [$M]", round(price_pred[1], 2), "Lower Range", round(price_pred[2], 2), "Upper Range", round(price_pred[3], 2), sep="\n")

  })
  
  output$plot2 <- renderPlot({boxplot(re_boston$price, main='Condo Price Boxplot',col='Red')})
  output$plot3 <- renderPlot({boxplot(re_boston$bathroom_number, main='Condo Bathroom Number Boxplot',col='Red')})
  output$plot4 <- renderPlot({boxplot(re_boston$bedroom_number, main='Condo Bedroom Number Boxplot',col='Red')})
  
  output$plot5 <- renderPlot({corrplot(cor(re_boston_ma, use="complete.obs"))})
  output$plot6 <- renderPlot(replayPlot(price_space_plot), width = 2750)
  output$plot7 <- renderPlot(replayPlot(pred_ref_plot), width = 2750)
  
  output$about_text = renderText(about_text_file)


}
