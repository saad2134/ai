function calculator()
    println("Simple Calculator in Julia")
    println("Operations: +, -, *, /, ^ (power), % (modulo)")
    
    while true
        print("Enter expression (or 'quit' to exit): ")
        input = readline()
        
        input == "quit" && break
        
        try
            # Evaluate the mathematical expression
            result = eval(Meta.parse(input))
            println("Result: $result")
        catch e
            println("Error: Invalid expression. Please try again.")
        end
    end
end

# Run the calculator
calculator()